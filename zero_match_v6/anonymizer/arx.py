# -*- coding: utf-8 -*-
"""Необязательное сравнение с ARX через внешнюю Java-обёртку (с ограничениями запуска)."""
import os
import shutil
import subprocess
import tempfile

import pandas as pd


def allowed_dirs():
    """Каталоги, из которых разрешено запускать jar (ZERO_MATCH_ARX_DIRS, через os.pathsep)."""
    env = os.environ.get("ZERO_MATCH_ARX_DIRS", "")
    dirs = [d for d in env.split(os.pathsep) if d] or [os.getcwd()]
    return [os.path.realpath(d) for d in dirs]


def arx_enabled():
    """ARX выключен, если задано ZERO_MATCH_DISABLE_ARX=1 (например, на общем сервере)."""
    return os.environ.get("ZERO_MATCH_DISABLE_ARX", "") != "1"


def is_allowed_jar(path):
    """Файл .jar внутри разрешённых каталогов (защита от запуска произвольного кода)."""
    if not path or not path.lower().endswith(".jar") or not os.path.isfile(path):
        return False
    real = os.path.realpath(path)
    return any(real == d or real.startswith(d + os.sep) for d in allowed_dirs())


def find_arx_jar():
    candidates = [os.environ.get("ARX_CLI_JAR", ""), "arx-cli.jar",
                  os.path.join(os.getcwd(), "arx-cli.jar"),
                  "/usr/local/lib/arx-cli.jar", "/opt/arx/arx-cli.jar"]
    return next((c for c in candidates if c and is_allowed_jar(c)), None)


def probe_arx_cli(jar_path, timeout=15):
    """Читает help и проверяет, что нужные флаги заявлены; ничего не предполагает."""
    if not arx_enabled():
        return {"ok": False, "message": "ARX отключён администратором", "help_text": ""}
    if not is_allowed_jar(jar_path):
        return {"ok": False, "message": "jar не найден или лежит вне разрешённых каталогов "
                                        "(ZERO_MATCH_ARX_DIRS)", "help_text": ""}
    if shutil.which("java") is None:
        return {"ok": False, "message": "Java не найдена", "help_text": ""}
    outputs = []
    for flag in ("--help", "-h", "help"):
        try:
            proc = subprocess.run(["java", "-jar", jar_path, flag], capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "message": f"help не ответил за {timeout} с", "help_text": ""}
        except Exception as exc:
            return {"ok": False, "message": f"Не удалось проверить CLI: {exc}", "help_text": ""}
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        outputs.append(out)
        if out.strip():
            break
    help_text = "\n".join(outputs).strip()
    if not help_text:
        return {"ok": False, "message": "CLI не вывел help; интерфейс не подтверждён", "help_text": ""}
    required = ["--input", "--output", "--quasi-identifiers", "--anonymization-type", "--k"]
    missing = [f for f in required if f not in help_text]
    if missing:
        return {"ok": False, "message": "Не подтверждены аргументы: " + ", ".join(missing),
                "help_text": help_text}
    return {"ok": True, "message": "Базовые аргументы найдены в help (нужен smoke-test)",
            "help_text": help_text}


def run_arx_cli(df, cols, k, l, jar_path, timeout=120):
    """(anonymized_df | None, error | None). В ARX уходят ТОЛЬКО QI-колонки."""
    probe = probe_arx_cli(jar_path, timeout=min(timeout, 15))
    if not probe["ok"]:
        return None, "ARX CLI не подтверждён: " + probe["message"]
    help_text = probe["help_text"]
    if l and l > 1 and "--l" not in help_text and "l-diversity" not in help_text.lower():
        return None, "CLI не заявляет l-diversity; запуск отменён, чтобы не подменять модель."
    if "--suppression-limit" not in help_text:
        return None, "CLI не заявляет suppression-limit; запуск отменён."
    tmp_dir = tempfile.mkdtemp(prefix="arx_")
    inp, out = os.path.join(tmp_dir, "input.csv"), os.path.join(tmp_dir, "output.csv")
    try:
        df[cols].to_csv(inp, index=False, encoding="utf-8")
        cmd = ["java", "-jar", jar_path, "--input", inp, "--output", out,
               "--quasi-identifiers", ",".join(cols), "--anonymization-type", "KANONYMITY",
               "--k", str(k), "--suppression-limit", "0.1"]
        if l and l > 1:
            cmd += ["--l", str(l)]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if res.returncode != 0:
            err = (res.stderr or res.stdout or "неизвестная ошибка ARX").strip()
            return None, f"ARX завершился с кодом {res.returncode}: {err[:500]}"
        if not os.path.isfile(out):
            return None, "ARX не создал выходной файл."
        return pd.read_csv(out), None
    except subprocess.TimeoutExpired:
        return None, f"ARX не уложился в {timeout} секунд."
    except Exception as e:
        return None, f"Ошибка при вызове ARX: {e}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# -*- coding: utf-8 -*-
"""Отпечатки, журнал аудита запуска, текстовый отчёт, экспорт (CSV/XLSX/Parquet/PDF)."""
import hashlib
import io
import json
import os
import platform
import uuid
from datetime import datetime

import numpy as np
import pandas as pd

from . import __version__
from .config import config_for_manifest

INTERNAL_BANNER = ("ВНУТРЕННИЙ ДОКУМЕНТ. Не передавать получателю данных и третьим лицам; хранить отдельно от "
                   "обезличенных данных (Приказ РКН № 140, Прил. 1, п. 2(2), 2(6)).")
DISCLAIMER = ("Отчёт формируется автоматически и не является заключением о соответствии "
              "152-ФЗ или иным требованиям.")


# ---------- отпечатки ----------
def bytes_hash(data):
    return hashlib.sha256(data).hexdigest()


def canonical_fingerprint(df):
    """
    Единый отпечаток таблицы: sha256 от канонической CSV-сериализации (без индекса).
    Не зависит от внутренних dtype pandas и индекса строк. Для выгрузки в CSV
    используйте bytes_hash(csv_bytes) — это хеш ровно того файла, что получит получатель.
    """
    return bytes_hash(df.to_csv(index=False).encode("utf-8"))


def json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def arrow_safe(df):
    """Колонки object со смешанными типами -> строки (для st.dataframe / Parquet)."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            kinds = {type(v) for v in out[c].dropna().head(2000)}
            if len(kinds) > 1:
                out[c] = out[c].map(lambda v: v if pd.isna(v) else str(v))
    return out


# ---------- журнал запуска (JSON) ----------
def build_audit(res, cfg, file_name, source_hash):
    """Журнал одного запуска. Seed и локальные пути в журнал не попадают."""
    df = res["df_safe"]
    audit = {
        "classification": INTERNAL_BANNER,
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tool_version": __version__,
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "numpy": np.__version__},
        "source": {"file": file_name, "file_sha256": source_hash,
                   "table_fingerprint": res["source_fingerprint"], "rows": res["rows_in"]},
        "output": {"rows": len(df), "columns": df.columns.tolist(),
                   "release_csv_sha256": res["release_sha256"]},
        "config": config_for_manifest(cfg),
        "bin_steps": res["bin_steps"], "dp": res["dp_info"], "suppression": res["suppression"],
        "metrics": {"before": res["before"], "after": res["after"], "attack": res["attack"],
                    "residual_pii": res["residual"], "matches_total": res["matches_total"]},
        "storage_mode": res["storage_mode"],
        "status": res["status_code"], "checks": res["checks"],
        "export_blocked": res["export_blocked"], "export_block_reasons": res["export_block_reasons"],
    }
    return json.dumps(audit, ensure_ascii=False, indent=2, default=json_default)


# ---------- текстовый отчёт ----------
def build_certificate(res, cfg, file_name, source_hash):
    k = cfg.k_target
    before, after = res["before"], res["after"]
    line = "=" * 68
    j = lambda cols: ', '.join(cols) if cols else '—'  # noqa: E731

    if cfg.quasi_cols:
        risk = (f"- QI: {j(cfg.quasi_cols)}\n- Целевое k: {k}\n"
                f"- Средний риск: {before['avg']}% -> {after['avg']}%\n"
                f"- Макс. риск: {after['max']}%\n"
                f"- Минимальный размер группы: {after['min_k']}\n"
                f"- Уникальных (k=1): {after['unique_share']}%\n"
                f"- Записей в группах < {k}: {after['below_k_share']}%")
    else:
        risk = "- НЕ ПРОВЕРЕНО (QI не выбраны)"

    ld = res["ldiv"]
    ld_txt = "— (чувствительные колонки не выбраны)"
    if ld is not None and not ld.empty and "Мин. l" in ld.columns:
        ld_txt = "\n".join(f"- «{r['Колонка']}»: мин. l={r['Мин. l']} (цель {cfg.l_target}), "
                           f"макс. t={r['Макс. t']} (цель {cfg.t_target})"
                           for _, r in ld.dropna(subset=["Мин. l"]).iterrows()) or "—"

    at = res["attack"]
    at_txt = "—" if not at else (
        f"- Среднее число кандидатов: {at['avg_candidates']}\n"
        f"- Однозначно найдено: {at['unique_share']}% (из них верно: {at['correct_unique_share']}%)\n"
        f"- Ожидаемая доля верных сопоставлений: {at['expected_reid']}%")

    supp = res["suppression"]
    bias = res["bias"]
    bias_txt = "—"
    if bias is not None and not bias.empty and supp.get("dropped"):
        bias_txt = "\n".join(f"- {r['Колонка']}: TVD {r['TVD до/после']}, больше всего удалено из "
                             f"«{r['Больше всего удалено из']}» ({r['Удалено из неё (%)']}%)"
                             for _, r in bias.iterrows())
    dp = res["dp_info"]
    if cfg.noise_mode == "ldp" and dp:
        noise_txt = (f"Лаплас, ε={cfg.epsilon} на колонку (ε-LDP), суммарно ε={round(cfg.epsilon * len(dp), 3)} "
                     f"(последовательная композиция). Границы обрезки взяты из процентилей данных — "
                     f"формальные гарантии требуют границ, заданных вручную.")
    else:
        noise_txt = f"Шум Лапласа {cfg.noise_pct}% от значения (эвристика, без гарантий ДП)."

    checks = "\n".join(
        f"[{'OK' if c['passed'] else 'НЕТ'}]{'' if c['enforced'] else ' (не блокирует)'} "
        f"{c['name']}: {c['detail']}" for c in res["checks"]) or "—"
    if res["status_code"] in ("ok", "ok_warn"):
        concl = "Обязательные проверки пройдены по выбранным QI и чувствительным колонкам.\n" + DISCLAIMER
    elif res["status_code"] == "needs_work":
        concl = ("Часть обязательных проверок НЕ пройдена. Данные нельзя считать обезличенными;\n"
                 "усильте обобщение/подавление и повторите проверку.")
    else:
        concl = "Оценка риска не проводилась. Выводов об обезличенности нет."
    util = res["utility"]
    util_txt = "—"
    if util is not None and not util.empty:
        util_txt = "\n".join(f"- {r['Колонка']}: {r['Метрика']}={r['Значение']}" for _, r in util.iterrows())
        if res["corr_diff"] is not None:
            util_txt += f"\n- Средний |Δ корреляции|: {res['corr_diff']}"
    notes = "".join(f"\n! {n}" for n in res["notes"])
    block = ("ДА: " + "; ".join(res["export_block_reasons"])) if res["export_blocked"] else "НЕТ"

    return f"""{INTERNAL_BANNER}
{line}
ОТЧЁТ ОБ ОБЕЗЛИЧИВАНИИ ДАННЫХ (v{__version__})
{line}
Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Источник: {file_name} (sha256 файла {source_hash[:16] or '—'}…)
Записей: {res['rows_in']} -> {len(res['df_safe'])}
Выгрузка CSV sha256: {res['release_sha256']}
ГСЧ: {'фиксированный (отладка)' if cfg.seed is not None else 'случайный (системная энтропия)'}
Режим хранения: {res['storage_mode']}{'' if res['storage_mode'] == 'isolated' else ' — НЕ изолирован: исходные и обезличенные данные в одной сессии'}

МЕТОДЫ:
1. Токенизация (случайные ID, одинаковое значение -> одинаковый токен): {j(cfg.id_cols)}
2. Обобщение категорий (иерархия + редкие < {cfg.rare_min_count}): {j(cfg.cat_cols)}
3. Перестановка Саттоло + шум: {j(cfg.derange_cols)}
   {noise_txt}
   Значений, оставшихся в своей строке: {res['matches_total']}
4. Биннинг: {j(cfg.kanon_cols)}; шаги: {res['bin_steps'] or '—'}
5. Зачистка текста: {j(cfg.text_cols)}{' (+NER: ' + str(res['ner_backend']) + ')' if cfg.use_ner else ''}
6. Цепи Маркова: {cfg.markov_col}
7. Подавление малых групп: режим «{cfg.supp_mode}», замаскировано {supp['masked']}, удалено {supp['dropped']}
8. Строки выходной таблицы перемешаны, исходные индексы в выгрузку не попадают.{notes}

КОЛОНКИ БЕЗ ОБРАБОТКИ (выгружаются как есть): {j(res['untouched'])}

k-АНОНИМНОСТЬ:
{risk}

l-РАЗНООБРАЗИЕ / t-БЛИЗОСТЬ:
{ld_txt}

АТАКА СВЯЗЫВАНИЯ (знание исходных QI):
{at_txt}

СМЕЩЕНИЕ ИЗ-ЗА УДАЛЕНИЯ СТРОК:
{bias_txt}

ПОЛЕЗНОСТЬ (до/после):
{util_txt}

ПРОВЕРКИ:
{checks}

СТАТУС: {res['status_label']}
Экспорт CSV заблокирован: {block}

ОГРАНИЧЕНИЯ:
- Риск оценён только по выбранным QI; другие колонки не учитывались.
- Независимые перестановки колонок разрушают связи между колонками.
- Детекторы ПДн (регулярки, контрольные суммы, опционально NER) могут ошибаться в обе стороны.
- Имена в тексте удаляются по значениям колонок токенизации (с падежами) и, при включении, NER.

ЗАКЛЮЧЕНИЕ: {concl}
{line}"""


# ---------- экспорт ----------
def to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        arrow_safe(df).to_excel(w, index=False, sheet_name="data")
    return buf.getvalue()


def to_parquet_bytes(df):
    buf = io.BytesIO()
    arrow_safe(df).to_parquet(buf, index=False)
    return buf.getvalue()


def find_font():
    cands = [os.environ.get("PDF_FONT", ""),
             "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
             "/usr/share/fonts/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
             "/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
             "C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/arial.ttf"]
    try:
        import matplotlib
        cands.append(os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf",
                                  "DejaVuSansMono.ttf"))
    except Exception:
        pass
    return next((p for p in cands if p and os.path.isfile(p)), None)


def build_pdf_bytes(text):
    """PDF из текста отчёта. Нужны reportlab и шрифт с кириллицей (или PDF_FONT=путь)."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfgen import canvas
    except ImportError as e:
        raise RuntimeError("Для PDF установите пакет reportlab.") from e
    font_path = find_font()
    if not font_path:
        raise RuntimeError("Не найден шрифт с кириллицей. Укажите путь в переменной PDF_FONT.")
    pdfmetrics.registerFont(TTFont("ReportFont", font_path))
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    size, lead, margin = 8, 10, 40
    max_w, y = width - 2 * margin, height - margin
    c.setFont("ReportFont", size)

    def wrap(line):
        if pdfmetrics.stringWidth(line, "ReportFont", size) <= max_w:
            return [line]
        out, cur = [], ""
        for word in line.split(" "):
            trial = (cur + " " + word).strip()
            if pdfmetrics.stringWidth(trial, "ReportFont", size) <= max_w:
                cur = trial
            else:
                out.append(cur)
                cur = word
        return out + [cur]

    for raw in text.split("\n"):
        for line in wrap(raw):
            if y < margin:
                c.showPage()
                c.setFont("ReportFont", size)
                y = height - margin
            c.drawString(margin, y, line)
            y -= lead
    c.save()
    return buf.getvalue()

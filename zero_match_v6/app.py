# -*- coding: utf-8 -*-
"""
Zero-Match Enterprise Anonymizer v6 — интерфейс Streamlit.

Запуск:  pip install -r requirements.txt  &&  streamlit run app.py
Ядро — пакет anonymizer/ (без зависимости от Streamlit, покрыто тестами: python tests/run_tests.py).
Переменные окружения: ZERO_MATCH_HOME (каталог данных рабочего пространства),
ZERO_MATCH_ARX_DIRS (откуда разрешён запуск arx-cli.jar), ZERO_MATCH_DISABLE_ARX=1 (отключить ARX),
PDF_FONT (шрифт с кириллицей для PDF). Приложение не даёт заключения о соответствии 152-ФЗ.
"""
import copy
import hashlib
import json
from dataclasses import fields

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from anonymizer import signing
from anonymizer import workspace as ws
from anonymizer.acts import build_acts_bundle, recipient_note
from anonymizer.arx import allowed_dirs, arx_enabled, find_arx_jar, is_allowed_jar, probe_arx_cli
from anonymizer.benchmark import analytical_utility_benchmark, optimize_generalization
from anonymizer.config import (COLUMN_LIST_FIELDS, NONE_OPTION, Config, config_for_manifest,
                               config_from_json, config_to_json)
from anonymizer.demo import generate_demo_data
from anonymizer.dossier import (CONCLUSIONS, LEGAL_BASES, PD_CATEGORIES, SUBJECT_CATEGORIES, autofill_composition,
                                card_warnings, empty_card, normalize_card, sufficiency_draft, validate_card)
from anonymizer.evidence import build_manifest, manifest_pdf_bytes, verify_manifest
from anonymizer.isolation import purge_source, storage_mode
from anonymizer.inventory import PROFILES, build_enterprise_report, build_inventory, risk_assessment
from anonymizer.pii_detect import classify_columns, detect_pii, suggest_config
from anonymizer.pipeline import run_pipeline
from anonymizer.policy import SCENARIOS, build_policy_artifact, evaluate_policy
from anonymizer.report import (INTERNAL_BANNER, arrow_safe, build_pdf_bytes, canonical_fingerprint, to_excel_bytes,
                               to_parquet_bytes)
from anonymizer.text_clean import ner_backend
from anonymizer.transforms import to_numeric_view

NOISE_LABELS = {"pct": "% от значения (эвристика)", "ldp": "Лаплас с ε (локальная ДП)"}
SUPP_LABELS = {"none": "не подавлять", "mask": "маскировать QI («*»)", "drop": "удалять строки"}
ss = st.session_state


def _stretch_kwargs():
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        return {"use_container_width": True}
    return {"width": "stretch"} if (major, minor) >= (1, 50) else {"use_container_width": True}


STRETCH = _stretch_kwargs()


def show_df(df, **kw):
    st.dataframe(arrow_safe(df) if isinstance(df, pd.DataFrame) else df, **STRETCH, **kw)


def show_fig(fig):
    st.plotly_chart(fig, **STRETCH)


@st.cache_resource(show_spinner=False)
def get_signer():
    """Существующий ключ подписи (не создаётся автоматически)."""
    return ws.default_signer()


# ================== СОСТОЯНИЕ ==================
def apply_values(values):
    for k, v in values.items():
        if k == "seed":
            ss["fixed_seed"] = v is not None
            if v is not None:
                ss["seed_value"] = int(v)
        elif v is not None:
            ss[k] = v


def apply_suggestions(sugg):
    vals = {k: v for k, v in sugg.items() if k != "markov_col"}
    vals["markov_col"] = sugg.get("markov_col") or NONE_OPTION
    apply_values(vals)


def init_state():
    defaults = Config()
    for f in fields(Config):
        if f.name != "seed":
            ss.setdefault(f.name, copy.deepcopy(getattr(defaults, f.name)))
    for k, v in {"fixed_seed": False, "seed_value": 42, "use_demo": True, "demo_size": 500,
                 "profile": "Analytics Balanced", "isolated_mode": False, "_uploader_gen": 0,
                 "bm_inline": False}.items():
        ss.setdefault(k, v)
    if "_arx_init" not in ss:
        ss["_arx_init"] = True
        ss["arx_jar"] = find_arx_jar() or "arx-cli.jar"
    pending = ss.pop("_pending_values", None)       # значения из кнопок «Применить…» (до создания виджетов)
    if pending:
        apply_values(pending)


def sanitize_state(all_cols):
    for k in COLUMN_LIST_FIELDS:
        ss[k] = [c for c in ss.get(k, []) if c in all_cols]
    if ss.get("markov_col") not in [NONE_OPTION] + all_cols:
        ss["markov_col"] = NONE_OPTION


def cfg_from_state():
    kwargs = {f.name: ss[f.name] for f in fields(Config) if f.name != "seed"}
    kwargs["seed"] = int(ss["seed_value"]) if ss["fixed_seed"] else None
    return Config(**kwargs)


def clear_session():
    for k in ("result", "_export_cache", "_data_sig", "_pii_report", "_classification", "_inventory",
              "_suggest", "_enterprise_risk", "_manifest", "gate_project"):
        ss.pop(k, None)
    st.cache_data.clear()


# ================== ДАННЫЕ ==================
@st.cache_data(show_spinner=False)
def cached_demo(n):
    return generate_demo_data(n)


def load_uploaded(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(uploaded)
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded)
        except UnicodeDecodeError:
            uploaded.seek(0)
            return pd.read_csv(uploaded, encoding="cp1251")
    return pd.read_excel(uploaded)


def analyze_data(df, sig):
    """Тяжёлый анализ колонок — один раз на набор данных, а не на каждый клик."""
    if ss.get("_data_sig") == sig:
        return
    ss["_data_sig"] = sig
    with st.spinner("Анализ колонок…"):
        pii = detect_pii(df)
        cl = classify_columns(df)
    ss["_pii_report"], ss["_classification"] = pii, cl
    ss["_inventory"] = build_inventory(df, cl)
    ss["_suggest"] = suggest_config(df, pii, cl)
    ss.pop("_enterprise_risk", None)
    apply_suggestions(ss["_suggest"])


# ================== ГРАФИКИ ==================
def build_pareto_figure(p):
    ours = p["df"][p["df"]["Движок"] == "Наш"].sort_values("Шаг_число")
    fig = px.line(ours, x="Средний риск (%)", y="Потеря информации (%)", markers=True, text="Шаг",
                  title=f"Компромисс: обобщение «{p['target_col']}» против риска")
    fig.update_traces(textposition="top center")
    arx = p["df"][p["df"]["Движок"] == "ARX"]
    if not arx.empty:
        r = arx.iloc[0]
        label = f"ARX (k={p['arx_k']}, l={p['arx_l']}, n={int(r['Строк'])})"
        fig.add_trace(go.Scatter(x=[r["Средний риск (%)"]], y=[r["Потеря информации (%)"]],
                                 mode="markers+text", text=[label], textposition="top center", name=label,
                                 marker=dict(size=22, color="purple", symbol="diamond",
                                             line=dict(color="black", width=2))))
    cur = ours[ours["Пресет"]]
    if not cur.empty:
        fig.add_trace(go.Scatter(x=cur["Средний риск (%)"], y=cur["Потеря информации (%)"],
                                 mode="markers+text", text=[f"выбран шаг {p['current_step']}"],
                                 textposition="bottom center", name=f"Выбранный шаг ({p['current_step']})",
                                 marker=dict(size=22, color="red", symbol="star",
                                             line=dict(color="black", width=2))))
    fig.update_layout(height=480, hovermode="closest")
    return fig


def distribution_figure(dist, col):
    """График по агрегатам, посчитанным при запуске: исходная таблица для него не нужна."""
    d, fig = dist[col], go.Figure()
    if d["type"] == "numeric":
        fig.add_trace(go.Bar(x=d["centers"], y=d["before"], name="До", opacity=0.6))
        fig.add_trace(go.Bar(x=d["centers"], y=d["after"], name="После (середины интервалов)", opacity=0.6))
        fig.update_layout(barmode="overlay")
    else:
        fig.add_trace(go.Bar(x=d["labels"], y=d["before"], name="До"))
        fig.add_trace(go.Bar(x=d["labels"], y=d["after"], name="После"))
        fig.update_layout(barmode="group")
    fig.update_layout(height=380, title=f"Распределение «{col}»")
    return fig


# ================== РЕЛИЗ-ГЕЙТ ==================
def release_allowed(res):
    """Экспорт разрешён, если нет блокирующих проверок и (при включённом гейте) выгрузка утверждена."""
    reasons = list(res["export_block_reasons"]) if res["export_blocked"] else []
    pid = ss.get("gate_project")
    if pid:
        try:
            status = ws.release_status(ws.default_paths()["db"], pid, res["release_sha256"])
            if not status["approved"]:
                reasons.append("этот файл (sha256) не утверждён в проекте — release gate")
        except Exception as e:  # noqa: BLE001
            reasons.append(f"не удалось проверить согласование: {e}")
    return not reasons, reasons


# ================== РЕЗУЛЬТАТЫ ==================
def tab_audit(res):
    k, before, after = res["k_target"], res["before"], res["after"]
    c = st.columns(6)
    c[0].metric("Строк", f"{len(res['df_safe'])} / {res['rows_in']}")
    c[1].metric("Средний риск", f"{after['avg']}%", delta=f"{round(after['avg'] - before['avg'], 2)}%",
                delta_color="inverse")
    c[2].metric("Макс. риск", f"{after['max']}%")
    c[3].metric("Мин. группа", after["min_k"])
    c[4].metric(f"Записей в группах < {k}", f"{after['below_k_share']}%")
    c[5].metric("Остаточные ПДн", res["residual"]["total"])
    st.markdown("**Проверки** (🔒 — обязательные, блокируют экспорт)")
    for ch in res["checks"]:
        mark = "✅" if ch["passed"] else ("❌" if ch["enforced"] else "⚠️")
        st.write(f"{mark} {'🔒 ' if ch['enforced'] else ''}{ch['name']} — {ch['detail']}")
    s = res["suppression"]
    st.caption(f"Подавление: «{SUPP_LABELS.get(s['mode'], s['mode'])}», замаскировано {s['masked']}, "
               f"удалено {s['dropped']}. Значений, оставшихся в своей строке: {res['matches_total']}.")
    bias = res["bias"]
    if bias is not None and not bias.empty and s["dropped"]:
        st.markdown("**Смещение из-за удаления строк**")
        show_df(bias)
    if res["opt_report"]:
        st.markdown("**Автоподбор шагов биннинга**")
        show_df(pd.DataFrame(res["opt_report"]))
    if res["dp_info"]:
        st.markdown("**Параметры ε-LDP**")
        show_df(pd.DataFrame(res["dp_info"]).T)
    if res["residual"]["by_column"]:
        st.warning(f"Остаточные ПДн по колонкам: {res['residual']['by_column']}")
    if res["residual"]["low_by_column"]:
        st.info(f"Похоже на номера без контрольной суммы (предупреждение): {res['residual']['low_by_column']}")
    if res["untouched"]:
        st.info("Без обработки выгружены: " + ", ".join(res["untouched"]))


def tab_klt(res):
    show_fig(px.bar(res["group_dist"], x="Размер группы", y="Записей (%)",
                    title="Доля записей по размеру группы (по QI)"))
    if res["ldiv"] is None or res["ldiv"].empty:
        st.info("Чувствительные колонки не выбраны — l и t не считались.")
    else:
        show_df(res["ldiv"])
        st.caption("l — число различных значений в группе; t — отличие распределения группы от общего "
                   "(EMD для порядковых, TVD для категориальных).")


def tab_attack(res):
    a = res["attack"]
    if not a:
        st.info("Выберите QI, чтобы смоделировать атаку.")
        return
    c = st.columns(4)
    c[0].metric("Среднее число кандидатов", a["avg_candidates"])
    c[1].metric("Найдено однозначно", f"{a['unique_share']}%")
    c[2].metric("…из них верно", f"{a['correct_unique_share']}%")
    c[3].metric("Ожидаемый успех", f"{a['expected_reid']}%")
    st.caption("Атакующий знает исходные QI человека и ищет строки в выгрузке. «Ожидаемый успех» — "
               f"вероятность верно выбрать запись среди кандидатов. Удалено при подавлении: {a['dropped_share']}%.")


def tab_utility(res):
    u = res["utility"]
    if u is None or u.empty:
        st.info("Нет колонок для сравнения.")
        return
    show_df(u)
    if res["corr_diff"] is not None:
        st.metric("Средний |Δ корреляции| числовых колонок", res["corr_diff"])
    col = st.selectbox("Колонка для графика", u["Колонка"].tolist(), key="util_col")
    if col in res["dist"]:
        show_fig(distribution_figure(res["dist"], col))


def _candidate_columns(res):
    cfg = res["_cfg"]
    handled = (set(cfg.id_cols) | set(cfg.derange_cols) | set(cfg.kanon_cols) | set(cfg.cat_cols)
               | set(cfg.text_cols) | ({cfg.markov_col} if cfg.markov_col != NONE_OPTION else set()))
    return [c for c in res["df_raw"].columns if c not in handled], [c for c in res["df_raw"].columns
                                                                    if c not in cfg.id_cols + cfg.text_cols]


def show_benchmark_result(b):
    c = st.columns(3)
    c[0].metric("Оценка", b["level"])
    c[1].metric(f"Потеря {b['primary_metric']}", f"{b['relative_loss_pct']}%")
    c[2].metric("Строк в сравнении", f"{b['rows_used']} ({b['coverage_pct']}%)")
    show_df(pd.DataFrame({"Исходные": b["original"], "Выгрузка": b["anonymized"]}).T)
    if b["note"]:
        st.warning(b["note"])


def tab_benchmark(res):
    if res.get("df_raw") is None:
        if res.get("benchmark"):
            show_benchmark_result(res["benchmark"])
            st.caption("Результат получен при запуске, до удаления исходных данных.")
        else:
            st.info("Исходные данные удалены из сессии (изолированный режим), бенчмарк недоступен. "
                    "Включите «ML-бенчмарк при запуске» до запуска или используйте режим разработки на демо-данных.")
        return
    targets, feats_all = _candidate_columns(res)
    st.caption("Парное сравнение: одна модель на исходных данных, одна на выгрузке (те же строки и "
               "разбиение). Цель должна быть колонкой, которую конвейер не меняет.")
    if not targets:
        st.info("Нет необработанных колонок для цели. Оставьте одну колонку вне шагов конвейера.")
        return
    target = st.selectbox("Целевая колонка", targets, key="bm_target")
    feats = st.multiselect("Признаки", [c for c in feats_all if c != target],
                           default=[c for c in res["quasi_cols"] if c != target], key="bm_features")
    trees = st.slider("Деревьев в RandomForest", 20, 200, step=20, key="bm_trees")
    if st.button("Запустить бенчмарк", key="bm_run"):
        try:
            with st.spinner("Обучение моделей…"):
                res["benchmark"] = analytical_utility_benchmark(
                    res["df_raw"], res["df_safe"], res["pair_pos"], target, feats, n_estimators=trees)
        except Exception as e:  # noqa: BLE001
            res["benchmark"] = None
            st.error(str(e))
    if res.get("benchmark"):
        show_benchmark_result(res["benchmark"])


def tab_optimizer(res):
    if res.get("df_raw") is None:
        st.info("Оптимизатор работает с исходной таблицей, поэтому недоступен после её удаления (изолированный "
                "режим). Подберите параметры на демо-данных или в режиме разработки и перенесите рекомендацию.")
        return
    targets, feats_all = _candidate_columns(res)
    st.caption("Оценивает только обобщение QI (шаг биннинга, редкие категории). Шум, подавление и "
               "перестановки в оценку не входят; результат — рекомендация, а не гарантия.")
    if not targets or not res["quasi_cols"]:
        st.info("Нужны QI и необработанная целевая колонка.")
        return
    target = st.selectbox("Целевая колонка", targets, key="opt_target")
    feats = st.multiselect("Признаки", [c for c in feats_all if c != target],
                           default=[c for c in res["quasi_cols"] if c != target], key="opt_features")
    n = st.slider("Кандидатов (максимум)", 4, 24, step=2, key="opt_n")
    if st.button("Подобрать обобщение", key="opt_run"):
        bar = st.progress(0.0, text="Старт")
        try:
            res["optimizer"] = optimize_generalization(
                res["df_raw"], res["quasi_cols"], target, feats, k=res["k_target"], max_candidates=n,
                progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
        except Exception as e:  # noqa: BLE001
            res["optimizer"] = None
            st.error(str(e))
        finally:
            bar.empty()
    o = res.get("optimizer")
    if o:
        t = o["table"].copy()
        t["Шаги"] = t["Шаги"].map(lambda d: json.dumps(d, ensure_ascii=False))
        show_df(t)
        rec = o["recommended"]
        st.success(f"Рекомендация: шаги {rec['manual_steps']}, редкие категории < {rec['rare_min_count']}"
                   + ("" if rec["feasible"] else " (k достигнуть не удалось — потребуется подавление)"))
        if st.button("Применить рекомендацию", key="opt_apply"):
            ss["_pending_values"] = {"manual_steps_json": json.dumps(rec["manual_steps"], ensure_ascii=False),
                                     "rare_min_count": rec["rare_min_count"]}
            st.rerun()


def tab_pareto(res):
    p = res["pareto"]
    if p is None:
        st.info(f"ℹ️ {res['pareto_note']}")
        return
    if p["arx_error"]:
        st.warning(f"⚠️ ARX: {p['arx_error']}")
    show_fig(build_pareto_figure(p))
    st.caption("🔴 выбранный шаг, 🟣 ARX. Чем ниже и левее точка — тем лучше компромисс.")
    with st.expander("Числовые значения"):
        show_df(p["df"])


def tab_export(res):
    out = res["df_safe"]
    show_df(out.head(20))
    allowed, reasons = release_allowed(res)
    if not allowed:
        st.error("🚫 Экспорт данных заблокирован: " + "; ".join(reasons))
    st.info("Здесь — только данные и записка для получателя. Отчёты, акты и описание методов лежат на вкладке "
            "«Внутренние документы»: получателю их не передавайте и храните отдельно от данных.")
    st.caption(f"sha256 выгрузки CSV: `{res['release_sha256']}` · режим хранения: {res['storage_mode']}")
    c = st.columns(2)
    c[0].download_button("📥 CSV", res["csv_bytes"], "enterprise_safe_data.csv", "text/csv",
                         key="dl_csv", disabled=not allowed)
    c[1].download_button("✉️ Записка получателю (без методов)", recipient_note(res).encode("utf-8"),
                         "recipient_note.txt", "text/plain", key="dl_note", disabled=not allowed)
    st.markdown("**Другие форматы** (по запросу; тоже под блокировкой)")
    cache = ss.setdefault("_export_cache", {})
    specs = {"xlsx": ("Excel", lambda: to_excel_bytes(out), "enterprise_safe_data.xlsx",
                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
             "parquet": ("Parquet", lambda: to_parquet_bytes(out), "enterprise_safe_data.parquet",
                         "application/octet-stream")}
    for col, (key, (label, builder, fname, mime)) in zip(st.columns(2), specs.items()):
        with col:
            ck = (res["release_sha256"], key)
            if not allowed:
                st.button(f"Подготовить {label}", key=f"prep_{key}", disabled=True)
                continue
            if ck not in cache and st.button(f"Подготовить {label}", key=f"prep_{key}"):
                try:
                    cache[ck] = builder()
                except Exception as e:  # noqa: BLE001
                    st.error(f"{label}: {e}")
            if ck in cache:
                st.download_button(f"⬇️ {label}", cache[ck], fname, mime, key=f"dl_{key}")


def tab_internal(res):
    st.warning("🔒 " + INTERNAL_BANNER)
    cfg = res["_cfg"]
    card = ss.get("card") or empty_card()
    user = (ss.get("ws_user") or {}).get("username", "")
    issues = validate_card(card)
    if issues:
        st.info(f"Карточка обезличивания заполнена не полностью ({len(issues)} пунктов) — акты будут с пропусками. "
                "Заполните её в разделе «Карточка обезличивания».")
    c = st.columns(3)
    c[0].download_button("📜 Отчёт (TXT)", res["cert_text"].encode("utf-8"), "INTERNAL_report.txt",
                         "text/plain", key="dl_txt")
    c[1].download_button("🧾 Журнал запуска (JSON)", res["audit_json"].encode("utf-8"), "INTERNAL_run_audit.json",
                         "application/json", key="dl_audit")
    c[2].download_button("📚 Акты и учёт операции (ZIP)",
                         build_acts_bundle(card, cfg, res, user, ss.get("_source_file_hash", "")),
                         "INTERNAL_local_acts.zip", "application/zip", key="dl_acts")
    cache = ss.setdefault("_export_cache", {})
    ck = (res["release_sha256"], "pdf")
    if ck not in cache and st.button("Подготовить PDF-отчёт", key="prep_pdf"):
        try:
            cache[ck] = build_pdf_bytes(res["cert_text"])
        except Exception as e:  # noqa: BLE001
            st.error(f"PDF: {e}")
    if ck in cache:
        st.download_button("⬇️ PDF-отчёт", cache[ck], "INTERNAL_report.pdf", "application/pdf", key="dl_pdf")
    st.caption("Акты — черновики для юриста и ответственного за ПДн. В них отмечено, что требует решения оператора "
               "(например, соответствие алгоритма перемешивания формулировке Приложения № 1).")


def tab_policy(res, df_raw):
    cfg = res["_cfg"]
    st.markdown("**Инвентаризация и оценка исходных данных** (без значений из таблицы)")
    show_df(ss["_inventory"])
    if df_raw is None:
        st.caption("Повторная оценка исходных данных недоступна: они удалены из сессии.")
    elif st.button("Оценить риск исходных данных", key="ent_risk"):
        ss["_enterprise_risk"] = risk_assessment(df_raw, cfg.quasi_cols, cfg.k_target)
    risk = ss.get("_enterprise_risk")
    if risk:
        c = st.columns(3)
        c[0].metric("Уровень риска", risk["level"])
        c[1].metric("Баллы", risk["score"])
        c[2].metric("Ячеек с ПДн (high)", risk["pii_high_cells"])
        st.caption(risk["note"])

    st.divider()
    st.markdown("**Политика выпуска** — автоматические проверки по результату запуска + ручные подтверждения")
    scenario = st.selectbox("Сценарий", list(SCENARIOS), key="scenario")
    st.caption(SCENARIOS[scenario]["description"])
    approvals = {a: st.checkbox(a, key=f"att_{res['release_sha256'][:8]}_{scenario}_{a}")
                 for a in SCENARIOS[scenario]["attestations"]}
    ev = evaluate_policy(scenario, res["summary"], res.get("benchmark"), approvals, card=ss.get("card"))
    icon = {"READY_FOR_REVIEW": "✅", "NEEDS_ATTESTATION": "🟡", "BLOCKED": "🚫", "NO_RUN": "ℹ️"}[ev["decision"]]
    st.markdown(f"### {icon} {ev['decision']}")
    show_df(pd.DataFrame(ev["checks"]))
    st.caption("READY_FOR_REVIEW означает «можно отправлять на согласование», а не «разрешено к выпуску».")

    st.divider()
    st.markdown("**Пакет доказательств**")
    signer = get_signer()
    st.caption("Подпись Ed25519: " + (f"включена (key_id {signer.key_id})" if signer else
                                      "выключена (нет ключа — создаётся явно в разделе «Рабочее пространство»)"))
    if st.button("Сформировать манифест", key="mf_build"):
        ss["_manifest"] = build_manifest(res["summary"], config_for_manifest(cfg), ss.get("_source_file_hash", ""),
                                         build_policy_artifact(ev) if ev["decision"] != "NO_RUN" else None,
                                         res.get("benchmark"), signer)
    m = ss.get("_manifest")
    if m and m["release"]["csv_sha256"] == res["release_sha256"]:
        v = verify_manifest(m, signer.public_pem if signer else None)
        st.write(f"Хеш манифеста: {'✅' if v['hash_ok'] else '❌'} · цепочка: "
                 f"{'✅' if v['chain_ok'] else '❌'} · подпись: {v['signature']}")
        c = st.columns(3)
        c[0].download_button("🧾 Манифест (JSON)", json.dumps(m, ensure_ascii=False, indent=2).encode("utf-8"),
                             "evidence_manifest.json", "application/json", key="dl_mf")
        c[1].download_button("📋 Политика (JSON)", json.dumps(build_policy_artifact(ev), ensure_ascii=False,
                                                             indent=2, default=str).encode("utf-8"),
                             "policy_artifact.json", "application/json", key="dl_pol")
        try:
            c[2].download_button("📄 Манифест (PDF)", manifest_pdf_bytes(m), "evidence_manifest.pdf",
                                 "application/pdf", key="dl_mfpdf")
        except RuntimeError as e:
            c[2].caption(str(e))
    with st.expander("Проверить чужой манифест"):
        up = st.file_uploader("Манифест (JSON)", type=["json"], key="mf_upload")
        pem = st.text_area("Доверенный публичный ключ (PEM), необязательно", key="mf_pem")
        if up is not None:
            try:
                v = verify_manifest(json.loads(up.getvalue().decode("utf-8")), pem or None)
                st.json(v)
                if v["signature"] == "valid_untrusted":
                    st.warning("Подпись верна для ключа из самого манифеста; без закреплённого ключа "
                               "это не доказывает авторство.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Не удалось проверить: {e}")
    rep = build_enterprise_report(ss["_inventory"], risk, scenario, cfg, res["summary"])
    st.download_button("📊 Enterprise-отчёт (JSON)", json.dumps(rep, ensure_ascii=False, indent=2,
                                                               default=str).encode("utf-8"),
                       "enterprise_report.json", "application/json", key="dl_ent")


def render_results(res, signature, df_raw):
    st.divider()
    st.subheader("3. Результаты")
    if signature is not None and res["signature"] != signature:
        st.info("ℹ️ Данные или настройки изменились после запуска. Ниже прежний результат — "
                "нажмите «Запустить конвейер», чтобы обновить.")
    if res["storage_mode"] == "isolated":
        st.success("🔒 Изолированный режим: исходные данные и связь строк удалены из результата и сессии приложения.")
    else:
        st.warning("⚠️ Режим хранения не изолирован: исходные и обезличенные данные находятся в одной сессии. "
                   "Для реальных ПДн это не соответствует требованию об исключении совместного хранения.")
    for n in res["notes"]:
        st.warning(f"⚠️ {n}")
    st.markdown(f"### {res['status_label']}")
    if res["export_blocked"]:
        st.error("🚫 Экспорт заблокирован: " + "; ".join(res["export_block_reasons"]))
    tabs = st.tabs(["Аудит", "k / l / t", "Атака", "Полезность", "ML-бенчмарк", "Оптимизатор",
                    "Парето", "База и экспорт", "Внутренние документы", "Политики и доказательства"])
    for tab, fn in zip(tabs, (tab_audit, tab_klt, tab_attack, tab_utility, tab_benchmark, tab_optimizer,
                              tab_pareto, tab_export, tab_internal)):
        with tab:
            fn(res)
    with tabs[9]:
        tab_policy(res, df_raw)


CARD_TEXT = [("operator_name", "Оператор (наименование)"), ("responsible_person", "Ответственный за обезличивание"),
             ("dpo_contact", "Контакт ответственного за ПДн"), ("purpose", "Цель обработки обезличенных данных"),
             ("legal_basis_details", "Пояснение к правовому основанию"),
             ("source_location", "Где хранится исходный массив (система/контур)"),
             ("result_location", "Где хранятся обезличенные данные (система/контур)"),
             ("additional_info_location", "Где хранятся соответствия (токены)"),
             ("recipients", "Получатели обезличенных данных"), ("retention_period", "Срок хранения"),
             ("notes", "Примечания")]
CARD_COLS = ["column", "role", "treatment"]


def apply_card_to_state(card):
    c = normalize_card(card)
    for k, _ in CARD_TEXT:
        ss[f"card_{k}"] = c[k]
    ss["card_legal_basis"] = c["legal_basis"]
    ss["card_subject_categories"] = c["subject_categories"]
    ss["card_pd_categories"] = c["pd_categories"]
    ss["card_special"] = bool(c["special_categories_present"])
    for k in ("assessed_by", "assessed_on", "rationale"):
        ss[f"card_s_{k}"] = c["sufficiency"][k]
    ss["card_s_conclusion"] = c["sufficiency"]["conclusion"]
    ss["card_comp_base"] = pd.DataFrame(c["data_composition"], columns=CARD_COLS)
    ss["card_comp_gen"] = ss.get("card_comp_gen", 0) + 1


def card_section(cfg, res, df_raw):
    """Карточка обезличивания (Приказ РКН № 140, Прил. 1, п. 1(2), 1(3), 1(6)) — внутренний документ."""
    with st.expander("📋 Карточка обезличивания (Приказ РКН № 140)"):
        if "card_operator_name" not in ss:
            apply_card_to_state(ss.get("card") or empty_card())
        pending = ss.pop("_pending_card", None)
        if pending is not None:
            apply_card_to_state(pending)
        st.caption("Определите до обезличивания: состав ПДн, субъектов, основание и оцените достаточность метода. "
                   "Это внутренний документ; выводы делает человек, инструмент только подсказывает.")
        for key, label in CARD_TEXT[:4]:
            st.text_input(label, key=f"card_{key}")
        st.selectbox("Правовое основание", [""] + LEGAL_BASES, key="card_legal_basis",
                     format_func=lambda x: x or "— выберите —")
        st.text_input(CARD_TEXT[4][1], key="card_legal_basis_details")
        a, b = st.columns(2)
        a.multiselect("Категории субъектов", SUBJECT_CATEGORIES, key="card_subject_categories")
        b.multiselect("Категории ПДн", PD_CATEGORIES, key="card_pd_categories")
        st.checkbox("В массиве есть специальные/биометрические категории", key="card_special")
        st.markdown("**Состав обезличиваемых данных**")
        if st.button("Заполнить состав по инвентаризации", key="card_autofill"):
            ss["card_comp_base"] = pd.DataFrame(autofill_composition(ss["_inventory"], cfg), columns=CARD_COLS)
            ss["card_comp_gen"] = ss.get("card_comp_gen", 0) + 1
        comp = st.data_editor(ss["card_comp_base"], num_rows="dynamic", key=f"card_comp_{ss.get('card_comp_gen', 0)}")
        st.markdown("**Хранение**")
        for key, label in CARD_TEXT[5:]:
            st.text_input(label, key=f"card_{key}")
        st.markdown("**Оценка достаточности метода** (заполняет человек)")
        if st.button("Подставить черновик обоснования из метрик", key="card_draft", disabled=res is None):
            ss["card_s_rationale"] = sufficiency_draft(res["summary"], res["attack"])
        st.text_input("Кто провёл оценку (ФИО, должность)", key="card_s_assessed_by")
        st.text_input("Дата оценки (ГГГГ-ММ-ДД)", key="card_s_assessed_on")
        st.selectbox("Вывод", CONCLUSIONS, key="card_s_conclusion", format_func=lambda x: x or "— выберите —")
        st.text_area("Обоснование", key="card_s_rationale", height=120)

        card = empty_card()
        for key, _ in CARD_TEXT:
            card[key] = ss[f"card_{key}"]
        card.update(legal_basis=ss["card_legal_basis"], subject_categories=list(ss["card_subject_categories"]),
                    pd_categories=list(ss["card_pd_categories"]), special_categories_present=bool(ss["card_special"]),
                    data_composition=[{k: str(r[k]) for k in CARD_COLS} for r in comp.fillna("").to_dict("records")
                                      if str(r.get("column", "")).strip()])
        card["sufficiency"] = {"assessed_by": ss["card_s_assessed_by"], "assessed_on": ss["card_s_assessed_on"],
                               "conclusion": ss["card_s_conclusion"], "rationale": ss["card_s_rationale"]}
        ss["card"] = card
        issues, warns = validate_card(card), card_warnings(card)
        (st.success if not issues else st.warning)("Карточка заполнена." if not issues else
                                                   f"Не хватает: {len(issues)} пункт(ов)")
        for i in issues:
            st.write(f"• {i}")
        for w in warns:
            st.info(w)


# ================== РАБОЧЕЕ ПРОСТРАНСТВО ==================
def workspace_panel(res):
    paths = ws.default_paths()
    db = paths["db"]
    ws.init(db)
    signer = get_signer()
    pem = signer.public_pem if signer else None
    if not signing.crypto_available():
        st.caption("Пакет cryptography не установлен: журнал защищён только хеш-цепочкой (без подписей).")

    if not ws.has_users(db):
        tok_path = ws.ensure_bootstrap_token(paths["token"])
        st.info(f"Первичная настройка. Токен лежит на сервере в файле `{tok_path}` — прочитайте его там "
                "и введите ниже. Без токена администратора создать нельзя.")
        token = st.text_input("Токен первичной настройки", type="password", key="ws_boot_token")
        u = st.text_input("Логин администратора", key="ws_boot_user")
        p = st.text_input("Пароль (≥ 12 символов, заглавная буква и цифра)", type="password", key="ws_boot_pw")
        if st.button("Создать администратора", key="ws_boot_btn"):
            try:
                if signer is None and signing.crypto_available():     # явное действие: ключ до первого события
                    signer = signing.load_signer(paths["key"], create=True)
                    get_signer.clear()
                ws.bootstrap_admin(db, tok_path, token, u, p, signer)
                st.success("Администратор создан. Ключ подписи журнала создан рядом с базой — "
                           "сохраните открытый ключ (*.pub.pem) отдельно. Войдите.")
                st.rerun()
            except (PermissionError, ValueError) as e:
                st.error(str(e))
        return

    user = ss.get("ws_user")
    if not user:
        u = st.text_input("Логин", key="ws_login_user")
        p = st.text_input("Пароль", type="password", key="ws_login_pw")
        if st.button("Войти", key="ws_login_btn"):
            try:
                found = ws.authenticate(db, u, p, signer)
                if found:
                    ss["ws_user"] = found
                    st.rerun()
                else:
                    st.error("Неверный логин или пароль.")
            except ws.AccountLocked as e:
                st.error(str(e))
        return

    uid = user["id"]
    st.write(f"Вы вошли как **{user['username']}** ({user['role']})")
    if st.button("Выйти", key="ws_logout"):
        ss.pop("ws_user", None)
        ss.pop("gate_project", None)
        st.rerun()
    try:
        with st.expander("Сменить пароль"):
            old = st.text_input("Текущий пароль", type="password", key="ws_pw_old")
            new = st.text_input("Новый пароль", type="password", key="ws_pw_new")
            if st.button("Сменить", key="ws_pw_btn"):
                ws.change_password(db, uid, old, new, signer)
                st.success("Пароль изменён.")
        if user["role"] == "admin":
            workspace_admin(db, uid, signer, paths)
        workspace_projects(db, user, res, signer)
        workspace_audit(db, pem)
    except (PermissionError, ValueError) as e:
        st.error(str(e))


def workspace_admin(db, uid, signer, paths):
    with st.expander("Администрирование"):
        if signing.crypto_available() and signer is None:
            if st.button("Создать ключ подписи журнала (Ed25519)", key="ws_mkkey"):
                signing.load_signer(paths["key"], create=True)
                get_signer.clear()
                st.success(f"Ключ создан: {paths['key']} (приватный; открытый — рядом, *.pub.pem). "
                           "Сохраните открытый ключ отдельно — по нему проверяют подписи.")
                st.rerun()
        c = st.columns(3)
        nu = c[0].text_input("Логин", key="ws_new_user")
        npw = c[1].text_input("Временный пароль", type="password", key="ws_new_pw")
        nr = c[2].selectbox("Роль", list(ws.ROLES), key="ws_new_role")
        if st.button("Создать пользователя", key="ws_new_btn"):
            ws.create_user(db, uid, nu, npw, nr, signer)
            st.success("Пользователь создан.")
        users = ws.list_users(db, uid)
        show_df(pd.DataFrame(users))
        ids = [u["id"] for u in users]
        target = st.selectbox("Пользователь", ids, format_func=lambda i: next(u["username"] for u in users if u["id"] == i),
                              key="ws_act_user")
        a, b = st.columns(2)
        if a.button("Отключить", key="ws_deact"):
            ws.set_user_active(db, uid, target, False, signer)
            st.rerun()
        if b.button("Включить", key="ws_act"):
            ws.set_user_active(db, uid, target, True, signer)
            st.rerun()


def workspace_projects(db, user, res, signer):
    uid = user["id"]
    st.markdown("**Проекты**")
    if user["role"] in ("admin", "analyst"):
        c = st.columns([2, 3, 1])
        name = c[0].text_input("Название", key="ws_pr_name")
        purpose = c[1].text_input("Цель обработки", key="ws_pr_purpose")
        if c[2].button("Создать", key="ws_pr_btn"):
            ws.create_project(db, uid, name, purpose, signer)
            st.rerun()
    projects = ws.list_projects(db, uid)
    if not projects:
        st.info("Доступных проектов нет.")
        return
    pid = st.selectbox("Проект", [p["id"] for p in projects], key="ws_pr_sel",
                       format_func=lambda i: next(f"{p['name']} · {p['status']} · {p['owner']}"
                                                  for p in projects if p["id"] == i))
    runs = ws.list_runs(db, uid, pid)
    if runs:
        show_df(pd.DataFrame([{k: r[k] for k in ("id", "status", "actor", "release_sha256", "created_at")}
                              for r in runs]))
    if user["role"] in ("admin", "analyst"):
        c1, c2 = st.columns(2)
        if c1.button("Сохранить карточку в проект", key="ws_card_save"):
            sha = ws.save_card(db, uid, pid, ss.get("card") or empty_card(), signer)
            st.success(f"Карточка сохранена (sha256 {sha[:12]}…). Для согласования она должна быть заполнена.")
        if c2.button("Загрузить карточку проекта", key="ws_card_load"):
            got = ws.get_card(db, uid, pid)
            if got:
                ss["_pending_card"] = got["card"]
                st.rerun()
            st.info("У проекта пока нет карточки.")
    if res is not None and user["role"] in ("admin", "analyst"):
        if res["export_blocked"]:
            st.caption("Текущий результат заблокирован проверками — на согласование не отправляется.")
        elif st.button("Отправить текущий результат на согласование", key="ws_submit"):
            ws.submit_run(db, uid, pid, res["summary"], signer)
            st.success("Отправлено. Решение привязано к sha256 выгрузки.")
            st.rerun()
    if user["role"] in ("admin", "reviewer"):
        for r in [r for r in runs if r["status"] == "SUBMITTED"]:
            with st.container(border=True):
                s = r["summary"]
                st.write(f"Прогон #{r['id']} от {r['actor']} · sha256 `{r['release_sha256'][:16]}…`")
                st.json({k: s.get(k) for k in ("status_code", "storage_mode", "min_k", "k_target", "l_min", "t_max",
                                               "residual_high", "suppression_pct", "rows_in", "rows_out",
                                               "card_sha256")})
                with st.expander("Карточка обезличивания (снимок на момент отправки)"):
                    st.json(s.get("card", {}))
                comment = st.text_input("Комментарий (обязателен при отклонении)", key=f"ws_cm_{r['id']}")
                a, b = st.columns(2)
                if a.button("Утвердить", key=f"ws_ok_{r['id']}"):
                    ws.review_run(db, uid, r["id"], "APPROVED", comment, signer)
                    st.rerun()
                if b.button("Отклонить", key=f"ws_no_{r['id']}"):
                    ws.review_run(db, uid, r["id"], "REJECTED", comment, signer)
                    st.rerun()
    gate = st.checkbox("Release gate: разрешать экспорт только утверждённой в этом проекте выгрузки",
                       value=ss.get("gate_project") == pid, key=f"ws_gate_{pid}")
    if gate:
        ss["gate_project"] = pid
    elif ss.get("gate_project") == pid:
        ss.pop("gate_project", None)


def workspace_audit(db, pem):
    st.markdown("**Журнал аудита** (проверка только читает)")
    if st.button("Проверить журнал", key="ws_verify"):
        v = ws.verify_audit(db, pem)
        (st.success if v["valid"] else st.error)(
            f"{'Цепочка цела' if v['valid'] else 'ОШИБКА: ' + v['reason']} · событий {v['events']} · "
            f"режим: {v['trust']} · подписей {v['verified_signatures']}/{v['signed_events']}")
        if v["valid"] and v["trust"] != "signed-and-verified":
            st.warning("Не все события подписаны или ключ не передан: цепочка защищает только от "
                       "частичных правок. Для доверия закрепите открытый ключ отдельно от сервера.")
    if st.button("Подготовить пакет журнала", key="ws_bundle"):
        ss["_audit_bundle"] = json.dumps(ws.export_audit_bundle(db, pem), ensure_ascii=False, indent=2,
                                         default=str).encode("utf-8")
    if ss.get("_audit_bundle"):
        st.download_button("⬇️ Журнал (JSON)", ss["_audit_bundle"], "workspace_audit_bundle.json",
                           "application/json", key="dl_wsbundle")


# ================== ИНТЕРФЕЙС ==================
def main():
    st.set_page_config(page_title="Zero-Match Enterprise Anonymizer", page_icon="🛡️", layout="wide")
    st.title("🛡️ Zero-Match Enterprise Anonymizer")
    st.markdown("Локальный контур обезличивания (демо): k-анонимность, l-разнообразие, t-близость, "
                "подавление, шум (в т.ч. ε-LDP), атака связывания, ML-бенчмарк, политики и доказательства. "
                "_Не является заключением о соответствии 152-ФЗ._")
    init_state()

    with st.sidebar:
        st.header("📂 Данные")
        use_demo = st.checkbox("Загрузить демо-базу", key="use_demo")
        st.selectbox("Размер демо-базы", [6, 500, 5000], key="demo_size", disabled=not use_demo)
        st.divider()
        st.subheader("💾 Конфигурация")
        cfg_file = st.file_uploader("Загрузить конфигурацию (JSON)", type=["json"], key="cfg_uploader")

    if ss.get("_prev_demo") != use_demo:          # реальные данные -> изоляция включается сама
        ss["_prev_demo"] = use_demo
        ss["isolated_mode"] = not use_demo
    if use_demo:
        df_raw = cached_demo(ss["demo_size"])
        file_name = f"demo_{ss['demo_size']}"
        source_hash = canonical_fingerprint(df_raw)
    else:
        df_raw, file_name, source_hash = None, "uploaded", ""
        uploaded = st.file_uploader("Загрузите CSV, Excel или Parquet", type=["csv", "xlsx", "parquet"],
                                    key=f"data_upload_{ss['_uploader_gen']}")
        if uploaded is not None:
            file_name = uploaded.name
            source_hash = hashlib.sha256(uploaded.getvalue()).hexdigest()
            uploaded.seek(0)
            try:
                df_raw = load_uploaded(uploaded)
            except Exception as e:  # noqa: BLE001
                st.error(f"Не удалось прочитать файл: {e}")
                return
    if df_raw is None:
        res = ss.get("result")
        if res is not None and res.get("purged"):
            with st.sidebar:
                if st.button("🧹 Очистить сессию (результаты)", key="clear_session_p"):
                    clear_session()
                    st.rerun()
            st.success("🔒 Исходные данные удалены из сессии. Ниже результаты и документы; "
                       "для нового запуска загрузите файл.")
            render_results(res, None, None)
            card_section(res["_cfg"], res, None)
            st.divider()
            with st.expander("🏢 Рабочее пространство, согласование и журнал аудита"):
                workspace_panel(res)
        else:
            st.info("Загрузите файл или включите демо-базу в боковой панели.")
        return
    ss["_source_file_hash"] = source_hash
    if len(df_raw) > 500_000:
        st.warning("Большая таблица: обработка векторизована, но подавление, атака и бенчмарк займут время.")

    all_cols = df_raw.columns.tolist()
    data_sig = (file_name, df_raw.shape, tuple(all_cols))
    analyze_data(df_raw, data_sig)

    if cfg_file is not None:
        tok = (cfg_file.name, cfg_file.size)
        if ss.get("_cfg_tok") != tok:
            try:
                apply_values(config_from_json(cfg_file.getvalue().decode("utf-8")))
                ss["_cfg_tok"] = tok
                st.sidebar.success("Конфигурация загружена")
            except Exception as e:  # noqa: BLE001
                st.sidebar.error(f"Не удалось загрузить конфигурацию: {e}")
    sanitize_state(all_cols)

    handled_now = set()
    for key in ("id_cols", "derange_cols", "kanon_cols", "cat_cols", "text_cols"):
        handled_now |= set(ss[key])
    if ss["markov_col"] != NONE_OPTION:
        handled_now.add(ss["markov_col"])
    free_cols = [c for c in all_cols if c not in handled_now]
    if ss.get("bm_inline_target") not in free_cols:
        ss.pop("bm_inline_target", None)
    ss["bm_inline_features"] = [c for c in ss.get("bm_inline_features", []) if c in all_cols]

    with st.sidebar:
        st.divider()
        st.subheader("🔒 Хранение исходных данных")
        st.checkbox("Изолированный режим: удалять исходник после запуска", key="isolated_mode",
                    help="Приказ РКН № 140, Прил. 1, п. 1(6): исходные и обезличенные данные не хранятся совместно. "
                         "Для реальных данных включено автоматически; для демо можно выключить.")
        if not ss["isolated_mode"] and not use_demo:
            st.error("Режим разработки на реальных данных не соответствует требованию об исключении "
                     "совместного хранения. Результат нельзя отправить на согласование.")
        st.caption("Приложение убирает ссылки и кэши; полностью затереть память процесса Python не может — для "
                   "гарантий разделяйте контуры (отдельный хост/процесс).")
        with st.expander("ML-бенчмарк при запуске"):
            st.checkbox("Выполнить до удаления исходника", key="bm_inline", disabled=not free_cols)
            if free_cols:
                st.selectbox("Целевая колонка (не обрабатывается конвейером)", free_cols, key="bm_inline_target")
                st.multiselect("Признаки", all_cols, key="bm_inline_features")
            else:
                st.caption("Нет необработанных колонок для цели.")
        st.divider()
        st.subheader("🏷️ Профиль")
        st.selectbox("Профиль настроек", list(PROFILES), key="profile")
        if st.button("Применить профиль", key="apply_profile"):
            ss["_pending_values"] = dict(PROFILES[ss["profile"]])
            st.rerun()

        st.divider()
        st.subheader("🎛️ Шум")
        st.radio("Режим шума", list(NOISE_LABELS), format_func=NOISE_LABELS.get, key="noise_mode")
        ldp = ss["noise_mode"] == "ldp"
        st.slider("Шум (% от значения)", 0, 30, step=1, key="noise_pct", disabled=ldp)
        st.number_input("ε на колонку (меньше — больше шума)", min_value=0.1, step=0.5, key="epsilon",
                        disabled=not ldp)
        st.slider("Обрезка выбросов для ε-LDP (% с края)", 0.0, 5.0, step=0.5, key="dp_clip_pct",
                  disabled=not ldp)

        st.divider()
        st.subheader("🎯 Цели приватности")
        st.number_input("k (мин. размер группы)", min_value=2, step=1, key="k_target")
        st.number_input("l (мин. число разных значений)", min_value=1, step=1, key="l_target")
        st.slider("t (макс. отличие распределения)", 0.05, 1.0, step=0.05, key="t_target")
        st.checkbox("Нарушение l блокирует экспорт", key="enforce_l")
        st.checkbox("Нарушение t блокирует экспорт", key="enforce_t",
                    help="t-близость строгая и дорогая по полезности; по умолчанию — предупреждение.")

        st.divider()
        st.subheader("✂️ Обобщение и подавление")
        st.checkbox("Подбирать шаг биннинга автоматически", key="auto_step")
        st.text_input("Ручные шаги (JSON), приоритетнее автоподбора", key="manual_steps_json",
                      help='Например {"Возраст": 10}. Заполняется кнопкой в «Оптимизаторе».')
        st.selectbox("Малые группы (< k)", list(SUPP_LABELS), format_func=SUPP_LABELS.get, key="supp_mode")
        st.slider("Допустимая доля подавления (%)", 0.0, 30.0, step=0.5, key="supp_limit",
                  disabled=ss["supp_mode"] == "none")
        st.number_input("Редкие категории: меньше N → «Другое»", min_value=0, step=1, key="rare_min_count")
        st.checkbox("NER для имён в тексте", key="use_ner")
        st.text_input("Модель spaCy (если нет — natasha)", key="ner_model", disabled=not ss["use_ner"])
        if ss["use_ner"]:
            b = ner_backend(ss["ner_model"])
            st.caption(f"NER: {b}" if b else "NER недоступен — будут применены только шаблоны.")

        st.divider()
        st.checkbox("Фиксированный seed (только отладка)", key="fixed_seed",
                    help="Делает перестановку воспроизводимой — для реальных данных выключите. "
                         "Seed не сохраняется в конфигурацию и отчёты.")
        st.number_input("Seed", step=1, key="seed_value", disabled=not ss["fixed_seed"])

        with st.expander("⚖️ Сравнение с ARX"):
            if not arx_enabled():
                st.info("ARX отключён администратором (ZERO_MATCH_DISABLE_ARX=1).")
            else:
                st.checkbox("Наложить точку ARX на график", key="use_arx")
                st.text_input("Путь к arx-cli.jar", key="arx_jar",
                              help="Разрешены только .jar внутри: " + ", ".join(allowed_dirs()))
                if not is_allowed_jar(ss["arx_jar"]):
                    st.caption("Файл не найден или вне разрешённых каталогов (ZERO_MATCH_ARX_DIRS).")
                elif st.button("Проверить CLI", key="arx_probe"):
                    pr = probe_arx_cli(ss["arx_jar"])
                    (st.success if pr["ok"] else st.error)(pr["message"])
                st.number_input("ARX: k", min_value=2, step=1, key="arx_k")
                st.number_input("ARX: l", min_value=1, step=1, key="arx_l")
        st.info("Вычисления локальные. Исходная таблица хранится в памяти сессии для сравнения «до/после».")
        if st.button("🧹 Очистить сессию (результаты и исходные данные)", key="clear_session"):
            clear_session()
            st.rerun()

    # ---- основная область ----
    st.subheader("1. Исходная таблица")
    show_df(df_raw.head(20))
    st.caption(f"Всего строк: {len(df_raw)}")

    with st.expander("🔎 Автоопределение ПДн и роли колонок"):
        rep = ss.get("_pii_report")
        if rep is None or rep.empty:
            st.write("Явных ПДн по шаблонам не найдено.")
        else:
            show_df(rep)
        show_df(ss["_classification"])
        sugg = ss.get("_suggest", {})
        st.caption("Рекомендация: " + "; ".join(f"{k}: {', '.join(v) if isinstance(v, list) else v}"
                                                for k, v in sugg.items() if v))
        if st.button("Применить рекомендации", key="apply_sugg"):
            apply_suggestions(sugg)

    st.subheader("2. Настройка конвейера")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.multiselect("Токенизация (ФИО, ID):", all_cols, key="id_cols")
        st.multiselect("Перестановка + шум (числа):", all_cols, key="derange_cols")
    with c2:
        st.multiselect("Биннинг чисел (k-анонимность):", all_cols, key="kanon_cols")
        st.multiselect("Обобщение категорий:", all_cols, key="cat_cols")
    with c3:
        st.multiselect("Зачистка текста:", all_cols, key="text_cols")
        st.selectbox("Цепи Маркова:", [NONE_OPTION] + all_cols, key="markov_col")
    c4, c5 = st.columns(2)
    with c4:
        st.multiselect("QI для оценки риска:", all_cols, key="quasi_cols")
    with c5:
        st.multiselect("Чувствительные колонки (l, t):", all_cols, key="sensitive_cols")
    with st.expander("Иерархия категорий (JSON, необязательно)"):
        st.text_area('Пример: {"Город": {"Москва": "Центр", "Тверь": "Центр"}}', key="hierarchy_json", height=100)

    if not ss["quasi_cols"]:
        st.warning("⚠️ Не выбран ни один QI — проверка k, атака и Парето работать не будут.")
    handled = set()
    for key in ("id_cols", "derange_cols", "kanon_cols", "cat_cols", "text_cols"):
        handled |= set(ss[key])
    if ss["markov_col"] != NONE_OPTION:
        handled.add(ss["markov_col"])
    untouched = [c for c in all_cols if c not in handled]
    if untouched:
        st.info("ℹ️ Без обработки попадут в выгрузку как есть: " + ", ".join(untouched)
                + ". Прямые идентификаторы среди них блокируют экспорт.")

    cfg = cfg_from_state()
    signature = repr((data_sig, cfg))
    with st.sidebar:
        st.download_button("💾 Скачать конфигурацию (без seed)", config_to_json(cfg).encode("utf-8"),
                           "anonymizer_config.json", "application/json", key="dl_cfg")

    purge_now = False
    if st.button("🚀 Запустить конвейер", type="primary"):
        bar = st.progress(0.0, text="Старт")
        try:
            isolated = bool(ss["isolated_mode"])
            r = run_pipeline(df_raw, cfg, file_name, source_hash,
                             progress=lambda f, m: bar.progress(min(f, 1.0), text=m),
                             storage_mode=storage_mode(isolated))
            if ss.get("bm_inline") and ss.get("bm_inline_target"):
                try:
                    r["benchmark"] = analytical_utility_benchmark(
                        df_raw, r["df_safe"], r["pair_pos"], ss["bm_inline_target"],
                        ss.get("bm_inline_features") or r["quasi_cols"], n_estimators=60)
                except Exception as e:  # noqa: BLE001
                    r["notes"].append(f"ML-бенчмарк при запуске: {e}")
            r.update({"signature": signature, "_cfg": cfg})
            if isolated:
                purge_source(r)                      # исходник и pair_pos больше не хранятся вместе с результатом
                if not use_demo:
                    ss["_uploader_gen"] += 1         # новый ключ загрузчика сбрасывает файл из памяти виджета
                    purge_now = True
            ss["result"] = r
            ss.pop("_manifest", None)
        except Exception as e:  # noqa: BLE001
            st.error(f"Ошибка конвейера: {e}")
        finally:
            bar.empty()
    if purge_now:
        st.rerun()

    res = ss.get("result")
    card_section(cfg, res, df_raw)
    if res is not None:
        render_results(res, signature, df_raw)

    st.divider()
    with st.expander("🏢 Рабочее пространство, согласование и журнал аудита"):
        workspace_panel(res)


if __name__ == "__main__":
    main()

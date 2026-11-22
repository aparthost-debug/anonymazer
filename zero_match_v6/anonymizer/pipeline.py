# -*- coding: utf-8 -*-
"""Конвейер обезличивания. Не зависит от Streamlit."""
import json

import numpy as np
import pandas as pd

from .config import NONE_OPTION, Config, config_hash  # noqa: F401  (реэкспорт)
from .metrics import (compute_risk_metrics, distribution_summary, group_size_distribution,
                      linkage_attack, sensitive_diversity, suppression_bias, utility_report)
from .optimize import optimize_steps, pareto_analysis, suppress_small_groups
from .pii_detect import count_pii_cells, unhandled_direct_identifiers
from .report import (bytes_hash, build_audit, build_certificate, canonical_fingerprint)
from .rng import make_rngs
from .text_clean import clean_series, collect_names, ner_backend
from .transforms import (MarkovBehaviorGenerator, add_noise_ldp, add_noise_pct, auto_step,
                         generalize_categories, generalize_numeric_series, parse_hierarchy,
                         sattolo_derangement)

POS_COL = "_orig_pos"


def _is_num(s):
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _parse_steps(text, notes):
    if not text or not text.strip():
        return {}
    try:
        data = json.loads(text)
        return {str(k): int(v) for k, v in data.items() if int(v) >= 1}
    except (ValueError, TypeError, AttributeError) as e:
        notes.append(f"Ручные шаги биннинга проигнорированы: {e}")
        return {}


def run_pipeline(df_raw, cfg, file_name="data", source_hash="", progress=None,
                 storage_mode="unspecified"):
    """Возвращает словарь результатов (таблица, метрики, проверки, отчёт, журнал, сводка).
    storage_mode: "isolated" | "shared (dev)" — режим хранения исходных данных (см. isolation.py)."""
    prog = progress or (lambda frac, msg: None)
    rng, np_rng = make_rngs(cfg.seed)
    n, k = len(df_raw), cfg.k_target
    notes = []
    qi = [c for c in cfg.quasi_cols if c in df_raw.columns]

    df_safe = df_raw.copy()
    df_safe[POS_COL] = np.arange(n)

    # 1. Токенизация: одинаковые ID -> одинаковый случайный токен
    prog(0.05, "Токенизация")
    for col in cfg.id_cols:
        uniques = df_raw[col].dropna().unique().tolist()
        ids = list(range(1000, 1000 + len(uniques)))
        rng.shuffle(ids)
        mapping = {u: f"Client_{i}" for u, i in zip(uniques, ids)}
        df_safe[col] = df_raw[col].map(mapping)

    # 2. Обобщение категорий (иерархия + редкие значения)
    prog(0.12, "Обобщение категорий")
    try:
        hierarchy = parse_hierarchy(cfg.hierarchy_json)
    except ValueError as e:
        hierarchy = {}
        notes.append(f"Иерархия категорий проигнорирована: {e}")
    cat_maps = {}
    for col in cfg.cat_cols:
        if _is_num(df_raw[col]):
            notes.append(f"Колонка «{col}» числовая — обобщение категорий пропущено (используйте биннинг).")
            continue
        df_safe[col], cat_maps[col] = generalize_categories(
            df_safe[col], hierarchy.get(col), cfg.rare_min_count)

    # 3. Перестановка Саттоло + шум
    prog(0.2, "Перестановка и шум")
    matches_total, dp_info = 0, {}
    for col in cfg.derange_cols:
        shuffled = pd.Series(sattolo_derangement(df_safe[col].tolist(), rng), index=df_safe.index)
        if cfg.noise_mode == "ldp":
            shuffled, info = add_noise_ldp(shuffled, cfg.epsilon, np_rng, cfg.dp_clip_pct)
            if info:
                dp_info[col] = info
            else:
                notes.append(f"Колонка «{col}» не числовая — шум ε-LDP не применялся.")
        else:
            shuffled = add_noise_pct(shuffled, cfg.noise_pct, np_rng)
        matches_total += int((df_raw[col].astype(str) == shuffled.astype(str)).sum())
        df_safe[col] = shuffled

    # 4. Биннинг: ручные шаги > автоподбор > пресет
    prog(0.35, "Подбор обобщения")
    num_kanon = [c for c in cfg.kanon_cols if _is_num(df_raw[c])]
    for c in cfg.kanon_cols:
        if c not in num_kanon:
            notes.append(f"Колонка «{c}» не числовая — биннинг пропущен.")
    opt_report = []
    if cfg.auto_step and num_kanon:
        limit = cfg.supp_limit if cfg.supp_mode != "none" else 0.0
        bin_steps, opt_report = optimize_steps(df_safe, qi, num_kanon, k, limit)
    else:
        bin_steps = {c: auto_step(df_raw[c]) for c in num_kanon}
    bin_steps.update({c: s for c, s in _parse_steps(cfg.manual_steps_json, notes).items()
                      if c in num_kanon})
    for c, s in bin_steps.items():
        df_safe[c] = generalize_numeric_series(df_safe[c], s)

    # 5. Зачистка текста
    prog(0.5, "Зачистка текста")
    backend = None
    if cfg.text_cols:
        if cfg.use_ner:
            backend = ner_backend(cfg.ner_model)
            if backend is None:
                notes.append("NER недоступен (нет spaCy-модели и natasha) — использованы только шаблоны.")
        names = collect_names(df_raw, cfg.id_cols) if cfg.id_cols else None
        ner_arg = cfg.ner_model if (cfg.use_ner and backend) else None
        for col in cfg.text_cols:
            df_safe[col] = clean_series(df_safe[col], names, ner_arg)

    # 6. Цепи Маркова
    prog(0.6, "Цепи Маркова")
    if cfg.markov_col != NONE_OPTION:
        seqs = [str(v).split(' -> ') for v in df_raw[cfg.markov_col].dropna()]
        mk = MarkovBehaviorGenerator()
        mk.fit(seqs)
        starts = [s[0] for s in seqs if s]
        if starts:
            df_safe[cfg.markov_col] = [' -> '.join(mk.safe_generate(starts, rng)) for _ in range(n)]

    # 7. Подавление малых групп
    prog(0.7, "Подавление малых групп")
    df_safe, supp = suppress_small_groups(df_safe, qi, k, cfg.supp_mode)
    supp = {"mode": cfg.supp_mode, **supp}

    # 8. Перемешивание строк: порядок не должен выдавать исходные позиции
    order = list(range(len(df_safe)))
    rng.shuffle(order)
    df_safe = df_safe.iloc[order].reset_index(drop=True)

    # 9. Оценка
    prog(0.78, "Оценка риска")
    before = compute_risk_metrics(df_raw, qi, k)
    after = compute_risk_metrics(df_safe, qi, k)
    sens = [c for c in cfg.sensitive_cols if c in df_safe.columns and c not in qi]
    ldiv = sensitive_diversity(df_safe, qi, sens, cfg.l_target, cfg.t_target) if qi and sens else None
    group_dist = group_size_distribution(df_safe, qi)

    prog(0.85, "Атака связывания")
    attack = linkage_attack(df_raw, df_safe, qi, bin_steps, cat_maps, POS_COL) if qi else None
    pair_pos = df_safe[POS_COL].to_numpy().copy()      # секретная связь «строка выгрузки -> исходная»
    bias = suppression_bias(df_raw, qi, pair_pos) if qi else None

    out_df = df_safe.drop(columns=[POS_COL])
    prog(0.9, "Полезность и контроль ПДн")
    util_cols = [c for c in df_raw.columns if c not in cfg.id_cols and c not in cfg.text_cols]
    utility, corr_diff = utility_report(df_raw, out_df, util_cols)
    dist = distribution_summary(df_raw, out_df, util_cols)
    residual = count_pii_cells(out_df)

    handled = (set(cfg.id_cols) | set(cfg.derange_cols) | set(cfg.kanon_cols) | set(cfg.text_cols)
               | set(cat_maps) | ({cfg.markov_col} if cfg.markov_col != NONE_OPTION else set()))
    direct_unhandled = unhandled_direct_identifiers(out_df, handled)

    # 10. Проверки: enforced=True блокирует экспорт, False — только предупреждение
    checks = []

    def add(name, passed, detail, enforced=True):
        checks.append({"name": name, "passed": bool(passed), "detail": detail, "enforced": enforced})

    if qi:
        add("k-анонимность", after["below_k_share"] == 0,
            f"мин. группа {after['min_k']}, записей в группах < {k}: {after['below_k_share']}%")
        if ldiv is not None and not ldiv.empty and "Мин. l" in ldiv.columns:
            for _, r in ldiv.dropna(subset=["Мин. l"]).iterrows():
                add(f"l-разнообразие «{r['Колонка']}»", r["Записей в классах с l < L (%)"] == 0,
                    f"мин. l={int(r['Мин. l'])}, цель {cfg.l_target}", cfg.enforce_l)
                add(f"t-близость «{r['Колонка']}»", r["Записей в классах с t > T (%)"] == 0,
                    f"макс. t={r['Макс. t']}, цель {cfg.t_target}", cfg.enforce_t)
    add("Остаточные ПДн (высокая уверенность)", residual["total"] == 0,
        f"ячеек: {residual['total']} {residual['by_column'] or ''}".strip())
    add("Необработанные прямые идентификаторы", not direct_unhandled,
        ", ".join(direct_unhandled) if direct_unhandled else "нет")
    add("Подозрительные номера без контрольной суммы", residual["low_total"] == 0,
        f"ячеек: {residual['low_total']} {residual['low_by_column'] or ''}".strip(), enforced=False)

    failed_hard = [c for c in checks if c["enforced"] and not c["passed"]]
    failed_soft = [c for c in checks if not c["enforced"] and not c["passed"]]
    if not qi:
        status_code, status_label = "not_checked", "❌ НЕ ПРОВЕРЕНО (QI не выбраны)"
    elif failed_hard:
        status_code, status_label = "needs_work", "⚠️ Требуется доработка"
    elif failed_soft:
        status_code, status_label = "ok_warn", "✅ Обязательные проверки пройдены · ⚠️ есть замечания"
    else:
        status_code, status_label = "ok", "✅ Все проверки пройдены"

    reasons = [c["name"] for c in failed_hard]
    if not qi:
        reasons.append("не выбраны QI")
    if len(out_df) == 0:
        reasons.append("пустой результат")
    export_blocked = bool(reasons)

    # 11. Паретo
    prog(0.95, "Кривая Парето")
    pareto, pareto_note = None, None
    if not qi:
        pareto_note = "Выберите хотя бы один QI — без него кривая Парето не строится."
    elif not num_kanon:
        pareto_note = "Выберите числовую колонку в «Биннинг» (например, «Возраст»)."
    else:
        target = num_kanon[0]
        pdf, arx_err = pareto_analysis(
            df_raw, list(dict.fromkeys(qi + [target])), target, bin_steps[target], k=k,
            use_arx=cfg.use_arx, arx_jar=cfg.arx_jar, k_arx=cfg.arx_k, l_arx=cfg.arx_l)
        if pdf.empty:
            pareto_note = "Недостаточно данных для кривой Парето."
        else:
            pareto = {"df": pdf, "target_col": target, "current_step": bin_steps[target],
                      "arx_error": arx_err, "arx_k": cfg.arx_k, "arx_l": cfg.arx_l}

    untouched = [c for c in df_raw.columns if c not in handled]

    csv_bytes = out_df.to_csv(index=False).encode("utf-8-sig")
    release_sha256 = bytes_hash(csv_bytes)
    l_min = t_max = None
    if ldiv is not None and not ldiv.empty and "Мин. l" in ldiv.columns:
        v = ldiv.dropna(subset=["Мин. l"])
        if not v.empty:
            l_min, t_max = int(v["Мин. l"].min()), float(v["Макс. t"].max())
    n_supp = supp.get("masked", 0) + supp.get("dropped", 0)

    res = {
        "df_safe": out_df, "df_raw": df_raw, "rows_in": n, "before": before, "after": after,
        "group_dist": group_dist, "ldiv": ldiv, "attack": attack, "utility": utility,
        "corr_diff": corr_diff, "residual": residual, "checks": checks,
        "status_code": status_code, "status_label": status_label, "notes": notes,
        "untouched": untouched, "bin_steps": bin_steps, "opt_report": opt_report,
        "dp_info": dp_info, "suppression": supp, "matches_total": matches_total,
        "pareto": pareto, "pareto_note": pareto_note, "file_name": file_name,
        "k_target": k, "quasi_cols": qi, "bias": bias, "ner_backend": backend,
        "direct_unhandled": direct_unhandled, "export_blocked": export_blocked,
        "export_block_reasons": reasons, "csv_bytes": csv_bytes,
        "release_sha256": release_sha256, "pair_pos": pair_pos,
        "source_fingerprint": canonical_fingerprint(df_raw),
        "dist": dist, "storage_mode": storage_mode, "purged": False,
    }
    res["summary"] = {          # JSON-безопасная сводка для политик, доказательств и согласования
        "status_code": status_code, "rows_in": n, "rows_out": len(out_df),
        "k_target": k, "min_k": after["min_k"], "l_target": cfg.l_target, "l_min": l_min,
        "t_target": cfg.t_target, "t_max": t_max, "sensitive_evaluated": l_min is not None,
        "residual_high": residual["total"], "residual_low": residual["low_total"],
        "direct_unhandled": direct_unhandled, "suppression_pct": round(n_supp / max(n, 1) * 100, 2),
        "expected_reid": attack["expected_reid"] if attack else None,
        "quasi_cols": qi, "release_sha256": release_sha256,
        "source_fingerprint": res["source_fingerprint"], "config_hash": config_hash(cfg),
        "export_blocked": export_blocked, "checks": checks, "storage_mode": storage_mode,
    }
    res["cert_text"] = build_certificate(res, cfg, file_name, source_hash)
    res["audit_json"] = build_audit(res, cfg, file_name, source_hash)
    prog(1.0, "Готово")
    return res

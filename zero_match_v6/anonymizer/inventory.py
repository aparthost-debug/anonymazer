# -*- coding: utf-8 -*-
"""Инвентаризация данных и первичная оценка риска (без примеров значений)."""
import pandas as pd

from .config import config_for_manifest
from .metrics import compute_risk_metrics
from .pii_detect import classify_columns, count_pii_cells

# профили настроек (применяются по кнопке, ничего не скрывают)
PROFILES = {
    "Internal BI": {"k_target": 5, "l_target": 2, "t_target": 0.35, "noise_pct": 3,
                    "enforce_l": False, "enforce_t": False},
    "Analytics Balanced": {"k_target": 5, "l_target": 2, "t_target": 0.30, "noise_pct": 5,
                           "enforce_l": True, "enforce_t": False},
    "External Vendor": {"k_target": 10, "l_target": 3, "t_target": 0.25, "noise_pct": 8,
                        "enforce_l": True, "enforce_t": True},
    "Public Release": {"k_target": 20, "l_target": 3, "t_target": 0.15, "noise_pct": 10,
                       "enforce_l": True, "enforce_t": True},
}


def build_inventory(df, classification=None):
    """Роли, пропуски, уникальность, длины. Значения из данных сюда НЕ попадают."""
    cl = classify_columns(df) if classification is None else classification
    rows = []
    for _, r in cl.iterrows():
        s = df[r["column"]]
        item = {"column": r["column"], "dtype": r["dtype"], "role": r["suggested_role"],
                "confidence": r["confidence"], "reason": r["reason"],
                "missing_%": r["missing_pct"], "distinct": r["unique_count"],
                "distinct_%": round(r["unique_count"] / max(len(s), 1) * 100, 2)}
        if not pd.api.types.is_numeric_dtype(s):
            ln = s.dropna().astype(str).str.len()
            item["len_min"], item["len_max"] = (int(ln.min()), int(ln.max())) if len(ln) else (None, None)
        rows.append(item)
    return pd.DataFrame(rows)


def risk_assessment(df, qi_cols, k=5, sample_limit=5000):
    """Оценка исходных данных: риск по QI и ПДн (только счётчики, без значений)."""
    sample = df.head(sample_limit)
    pii = count_pii_cells(sample)
    qi = [c for c in qi_cols if c in df.columns]
    rm = compute_risk_metrics(df, qi, k) if qi else None
    score = 0
    if rm:
        score += 35 if rm["min_k"] < 2 else 25 if rm["min_k"] < 5 else 10 if rm["min_k"] < 10 else 0
        score += min(20, round(rm["unique_share"] / 100 * 40))
    score += 25 if pii["total"] else 0
    score += 5 if pii["low_total"] else 0
    level = "high" if score >= 60 else "medium" if score >= 30 else "low"
    return {"score": score, "level": level, "metrics": rm, "pii_high_cells": pii["total"],
            "pii_low_cells": pii["low_total"], "pii_columns": sorted(pii["by_column"]),
            "sample_rows": len(sample), "note": "PII и риск оценены по первым строкам таблицы."}


def build_enterprise_report(inventory, risk, profile, cfg, summary=None):
    return {"profile": profile, "inventory": inventory.to_dict(orient="records"), "risk": risk,
            "configuration": config_for_manifest(cfg), "pipeline": summary}

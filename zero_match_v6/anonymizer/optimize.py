# -*- coding: utf-8 -*-
"""Подбор обобщения, подавление малых групп, кривая Парето."""
import numpy as np
import pandas as pd

from .arx import run_arx_cli
from .metrics import compute_risk_metrics, equivalence_codes
from .transforms import auto_step, generalize_numeric_series, info_loss_numeric


def pareto_steps(current_step):
    """Шаги масштабируются от пресета, поэтому пресет всегда лежит на кривой."""
    mults = (0.1, 0.2, 0.5, 1, 1.5, 2, 3, 5)
    return sorted({max(1, int(round(current_step * m))) for m in mults})


def optimize_steps(df, qi, num_cols, k, max_suppress_pct, sample_limit=200_000):
    """
    Для каждой числовой QI-колонки выбирает минимальную потерю информации среди шагов,
    при которых доля записей в группах < k не превышает max_suppress_pct.
    Если ни один шаг не подходит — берёт шаг с наименьшей долей нарушений.
    Возвращает (steps, report).
    """
    steps = {c: auto_step(df[c]) for c in num_cols}
    work = df if len(df) <= sample_limit else df.sample(sample_limit, random_state=0)
    qi = [c for c in qi if c in df.columns]
    report = []
    for c in num_cols:
        if c not in qi:
            continue                      # шаг не влияет на k — оставляем пресет
        rows = []
        for s in pareto_steps(steps[c]):
            trial_steps = dict(steps)
            trial_steps[c] = s
            trial = work[qi].copy()
            for cc, ss in trial_steps.items():
                if cc in qi:
                    trial[cc] = generalize_numeric_series(trial[cc], ss)
            risk = compute_risk_metrics(trial, qi, k)
            rows.append((s, risk["below_k_share"], info_loss_numeric(work[c], trial[c])))
        feasible = [r for r in rows if r[1] <= max_suppress_pct]
        pick = (min(feasible, key=lambda r: (r[2], r[0])) if feasible
                else min(rows, key=lambda r: (r[1], r[2], r[0])))
        steps[c] = pick[0]
        report.append({"Колонка": c, "Выбранный шаг": pick[0],
                       "Групп < k (%)": pick[1], "Потеря информации (%)": pick[2],
                       "Условие выполнено": bool(feasible)})
    return steps, report


def suppress_small_groups(df, qi, k, mode):
    """
    mode='mask' — QI строк из малых групп заменяются на «*»; если «*»-группа сама
                  меньше k, эти строки удаляются.
    mode='drop' — такие строки удаляются.
    Возвращает (df, {'masked': n, 'dropped': n}).
    """
    info = {"masked": 0, "dropped": 0}
    qi = [c for c in qi if c in df.columns]
    if mode == "none" or not qi or len(df) == 0:
        return df, info
    codes = equivalence_codes(df, qi)
    bad = np.bincount(codes)[codes] < k
    n_bad = int(bad.sum())
    if n_bad == 0:
        return df, info
    if mode == "drop":
        info["dropped"] = n_bad
        return df.loc[~bad].reset_index(drop=True), info
    out = df.copy()
    out[qi] = out[qi].astype(object)
    out.loc[bad, qi] = "*"
    if n_bad < k:
        info["dropped"] = n_bad
        return out.loc[~bad].reset_index(drop=True), info
    info["masked"] = n_bad
    return out, info


def pareto_analysis(df, quasi_cols, target_col, current_step, k=5,
                    use_arx=False, arx_jar=None, k_arx=5, l_arx=2, sample_limit=50_000):
    """Возвращает (DataFrame, arx_error | None). Для больших таблиц считает по выборке."""
    work = df[quasi_cols]
    if len(work) > sample_limit:
        work = work.sample(sample_limit, random_state=0)
    rows = []
    for step in pareto_steps(current_step):
        t = work.copy()
        t[target_col] = generalize_numeric_series(t[target_col], step)
        risk = compute_risk_metrics(t, quasi_cols, k=k)
        rows.append({
            "Шаг": str(step), "Шаг_число": step,
            "Средний риск (%)": risk["avg"], "Макс. риск (%)": risk["max"],
            "Доля < k (%)": risk["below_k_share"],
            "Потеря информации (%)": info_loss_numeric(work[target_col], t[target_col]),
            "Движок": "Наш", "Строк": len(t), "Пресет": step == current_step,
        })

    arx_err = None
    if use_arx and arx_jar:
        arx_anon, arx_err = run_arx_cli(df, quasi_cols, k=k_arx, l=l_arx, jar_path=arx_jar)
        if arx_anon is not None:
            if all(c in arx_anon.columns for c in quasi_cols):
                r = compute_risk_metrics(arx_anon, quasi_cols, k=k)
                rows.append({
                    "Шаг": f"ARX (k={k_arx}, l={l_arx})", "Шаг_число": np.nan,
                    "Средний риск (%)": r["avg"], "Макс. риск (%)": r["max"],
                    "Доля < k (%)": r["below_k_share"],
                    "Потеря информации (%)": info_loss_numeric(df[target_col], arx_anon[target_col]),
                    "Движок": "ARX", "Строк": len(arx_anon), "Пресет": False,
                })
            else:
                arx_err = "В выходе ARX нет всех QI-колонок."
    return pd.DataFrame(rows), arx_err

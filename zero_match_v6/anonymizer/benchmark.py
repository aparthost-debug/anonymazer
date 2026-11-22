# -*- coding: utf-8 -*-
"""
Бенчмарк полезности для ML (scikit-learn, подключается лениво) и оптимизатор обобщения.

Сравнение «исходные -> обезличенные» парное: строки выгрузки сопоставляются с исходными по
секретному массиву pair_pos, который живёт только в памяти и не попадает ни в какие файлы.
"""
import itertools

import numpy as np
import pandas as pd

from .metrics import compute_risk_metrics
from .optimize import pareto_steps
from .transforms import auto_step, generalize_numeric_series


def _sk():
    try:
        import sklearn  # noqa: F401
    except ImportError as e:
        raise RuntimeError("Для бенчмарка установите scikit-learn.") from e


def align_released(df_safe, pair_pos):
    """Выгрузка с индексом = позиция исходной строки (только для парного сравнения)."""
    aligned = df_safe.copy()
    aligned.index = np.asarray(pair_pos)
    return aligned.sort_index()


def infer_task(y):
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15:
        return "regression"
    return "classification"


def _prep(X):
    X = X.copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(object).where(X[c].notna(), "__missing__").astype(str)
    return X


def _fit_score(Xtr, ytr, Xte, yte, task, n_estimators, seed):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, r2_score,
                                 roc_auc_score)
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    num = [c for c in Xtr.columns if pd.api.types.is_numeric_dtype(Xtr[c])]
    cat = [c for c in Xtr.columns if c not in num]
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    est = (RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
           if task == "classification"
           else RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1))
    pipe = Pipeline([("pre", pre), ("model", est)]).fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    if task == "regression":
        return {"R2": float(r2_score(yte, pred)), "MAE": float(mean_absolute_error(yte, pred))}
    out = {"Accuracy": float(accuracy_score(yte, pred)),
           "F1 (macro)": float(f1_score(yte, pred, average="macro"))}
    try:
        proba = pipe.predict_proba(Xte)
        classes = pipe.classes_
        out["ROC-AUC"] = float(roc_auc_score(yte, proba[:, 1]) if len(classes) == 2 else
                               roc_auc_score(yte, proba, multi_class="ovr", labels=classes))
    except Exception:
        pass
    return out


def _split(y, task, test_size, seed):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    strat = None
    if task == "classification":
        vc = y.value_counts()
        strat = y if (len(vc) > 1 and vc.min() >= 2) else None
    return train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)


def _primary(task):
    return "R2" if task == "regression" else "F1 (macro)"


def assess_loss(loss_pct):
    return ("Excellent" if loss_pct <= 5 else "Good" if loss_pct <= 15
            else "Moderate" if loss_pct <= 30 else "Poor")


def analytical_utility_benchmark(original, released, pair_pos, target, features, task=None,
                                 test_size=0.25, seed=42, n_estimators=100, sample_limit=20000):
    """
    Обучает одну модель на исходных данных и одну на выгрузке (те же строки и разбиение),
    сравнивает на отложенной части. Целевая переменная должна остаться неизменной.
    """
    _sk()
    anon = align_released(released, pair_pos)
    common = original.index.intersection(anon.index)
    if len(common) < 50:
        raise ValueError("Слишком мало общих строк для бенчмарка (нужно ≥ 50).")
    if target not in original.columns or target not in anon.columns:
        raise ValueError("Целевая колонка отсутствует в исходных данных или выгрузке.")
    features = [f for f in features if f in original.columns and f in anon.columns and f != target]
    if not features:
        raise ValueError("Нет общих признаков для бенчмарка.")
    if not original.loc[common, target].equals(anon.loc[common, target]):
        raise ValueError("Целевая колонка изменена конвейером — сравнение некорректно. "
                         "Выберите колонку, которая не обрабатывается.")
    if len(common) > sample_limit:
        common = pd.Index(np.random.default_rng(seed).choice(common, sample_limit, replace=False))
    y = original.loc[common, target]
    valid = y.notna().to_numpy()
    common, y = common[valid], y[valid]
    task = task or infer_task(y)
    Xo, Xa = _prep(original.loc[common, features]), _prep(anon.loc[common, features])
    tr, te = _split(y.reset_index(drop=True), task, test_size, seed)
    yv = y.to_numpy()
    mo = _fit_score(Xo.iloc[tr], yv[tr], Xo.iloc[te], yv[te], task, n_estimators, seed)
    ma = _fit_score(Xa.iloc[tr], yv[tr], Xa.iloc[te], yv[te], task, n_estimators, seed)
    key = _primary(task)
    loss = (mo[key] - ma[key]) / max(abs(mo[key]), 1e-9) * 100
    baseline = None
    if task == "classification":
        baseline = float(pd.Series(yv[tr]).value_counts(normalize=True).iloc[0])
    informative = not (task == "classification" and mo["Accuracy"] <= baseline + 0.02) \
        and not (task == "regression" and mo["R2"] <= 0.02)
    return {"task": task, "rows_used": int(len(common)), "coverage_pct": round(len(common) / len(original) * 100, 1),
            "original": mo, "anonymized": ma, "primary_metric": key,
            "relative_loss_pct": round(float(loss), 2), "level": assess_loss(loss),
            "baseline_accuracy": baseline, "informative": bool(informative),
            "note": None if informative else
            "Модель на исходных данных почти не лучше константы — оценка потери неинформативна."}


def optimize_generalization(df, qi_cols, target, features, k=5, task=None, max_candidates=12,
                            sample_limit=20000, n_estimators=60, seed=42, progress=None):
    """
    Перебирает шаги биннинга числовых QI и порог редких категорий. Для каждого кандидата —
    риск (по QI) и потеря полезности модели. Это оценка ТОЛЬКО обобщения QI:
    шум, подавление и перестановки в неё не входят.
    """
    _sk()
    if target in qi_cols:
        raise ValueError("Целевая колонка не должна быть QI.")
    work = df if len(df) <= sample_limit else df.sample(sample_limit, random_state=seed)
    work = work.reset_index(drop=True)
    qi = [c for c in qi_cols if c in work.columns]
    num_qi = [c for c in qi if pd.api.types.is_numeric_dtype(work[c])]
    cat_qi = [c for c in qi if c not in num_qi]
    feats = [f for f in features if f in work.columns and f != target]
    y = work[target]
    task = task or infer_task(y)
    tr, te = _split(y, task, 0.25, seed)
    yv = y.to_numpy()
    base = _fit_score(_prep(work[feats]).iloc[tr], yv[tr], _prep(work[feats]).iloc[te], yv[te],
                      task, n_estimators, seed)
    key = _primary(task)

    step_opts = [sorted({max(1, int(round(auto_step(work[c]) * m))) for m in (0.5, 1, 2, 4)})
                 for c in num_qi]
    rare_opts = [0, k, 3 * k] if cat_qi else [0]
    grid = [(dict(zip(num_qi, steps)), rare)
            for steps in itertools.product(*step_opts) for rare in rare_opts]
    if len(grid) > max_candidates:
        grid = [grid[i] for i in np.linspace(0, len(grid) - 1, max_candidates).astype(int)]

    rows = []
    for i, (steps, rare) in enumerate(grid):
        if progress:
            progress(i / max(len(grid), 1), f"Кандидат {i + 1}/{len(grid)}")
        cand = work.copy()
        for c, s in steps.items():
            cand[c] = generalize_numeric_series(cand[c], s)
        for c in cat_qi:
            if rare:
                counts = cand[c].value_counts()
                small = set(counts[counts < rare].index)
                cand[c] = cand[c].where(~cand[c].isin(small), "Другое")
        risk = compute_risk_metrics(cand, qi, k)
        Xc = _prep(cand[feats])
        sc = _fit_score(Xc.iloc[tr], yv[tr], Xc.iloc[te], yv[te], task, n_estimators, seed)
        loss = (base[key] - sc[key]) / max(abs(base[key]), 1e-9) * 100
        rows.append({"Шаги": steps, "Редкие < N": rare, "Средний риск (%)": risk["avg"],
                     "Записей в группах < k (%)": risk["below_k_share"], "Мин. группа": risk["min_k"],
                     f"Потеря {key} (%)": round(float(loss), 2)})
    res = pd.DataFrame(rows)
    loss_col = f"Потеря {key} (%)"
    res["Парето"] = [not ((res["Средний риск (%)"] <= r["Средний риск (%)"]) &
                          (res[loss_col] <= r[loss_col]) &
                          ((res["Средний риск (%)"] < r["Средний риск (%)"]) |
                           (res[loss_col] < r[loss_col]))).any() for _, r in res.iterrows()]
    feasible = res[res["Записей в группах < k (%)"] == 0]
    pool = feasible if not feasible.empty else res
    best = pool.sort_values([("Записей в группах < k (%)"), loss_col, "Средний риск (%)"]).iloc[0]
    return {"table": res, "task": task, "baseline": base, "primary_metric": key,
            "recommended": {"manual_steps": {c: int(s) for c, s in best["Шаги"].items()},
                            "rare_min_count": int(best["Редкие < N"]),
                            "feasible": bool(not feasible.empty)}}

# -*- coding: utf-8 -*-
"""Метрики: риск (k), l-разнообразие, t-близость, атака связывания, полезность."""
import numpy as np
import pandas as pd

from .transforms import generalize_numeric_series, to_numeric_view


def equivalence_codes(df, qi):
    """Номер класса эквивалентности для каждой строки. NaN — отдельное значение."""
    key = df[qi].astype(str).fillna("<NA>")
    return key.groupby(qi, sort=False, dropna=False).ngroup().to_numpy()


def compute_risk_metrics(df, quasi_identifiers, k=5):
    empty = {"avg": 0.0, "max": 0.0, "unique_share": 0.0, "below_k_share": 0.0, "min_k": 0}
    qi = [c for c in (quasi_identifiers or []) if c in df.columns]
    if not qi or len(df) == 0:
        return empty
    codes = equivalence_codes(df, qi)
    sizes = np.bincount(codes)[codes]
    risks = 1.0 / sizes
    return {
        "avg": round(float(risks.mean()) * 100, 2),
        "max": round(float(risks.max()) * 100, 2),
        "unique_share": round(float((sizes == 1).mean()) * 100, 2),
        "below_k_share": round(float((sizes < k).mean()) * 100, 2),
        "min_k": int(sizes.min()),
    }


def group_size_distribution(df, qi):
    """Доля записей по размеру группы (для гистограммы)."""
    labels = ["1", "2", "3", "4", "5–9", "10–19", "20+"]
    qi = [c for c in qi if c in df.columns]
    if not qi or df.empty:
        return pd.DataFrame({"Размер группы": labels, "Записей (%)": [0.0] * len(labels)})
    codes = equivalence_codes(df, qi)
    sizes = np.bincount(codes)[codes]
    cat = pd.cut(pd.Series(sizes), bins=[0, 1, 2, 3, 4, 9, 19, np.inf], labels=labels)
    dist = cat.value_counts(sort=False) / len(sizes) * 100
    return pd.DataFrame({"Размер группы": labels, "Записей (%)": dist.round(2).to_numpy()})


# ---------- l-разнообразие и t-близость ----------
def _encode_sensitive(s, bins=10, max_card=200):
    """-> (коды, порядковая?, число значений) или None, если колонка слишком разнообразна."""
    non_null = s.dropna()
    if non_null.empty:
        return None
    num = to_numeric_view(s)
    if num.notna().sum() >= 0.8 * len(non_null):
        try:
            b = pd.qcut(num, q=min(bins, max(2, int(num.nunique()))), duplicates="drop", labels=False)
        except ValueError:
            b = pd.Series(0, index=s.index)
        b = pd.Series(b, index=s.index)
        fill = (b.max() + 1) if b.notna().any() else 0
        codes = b.fillna(fill).astype(int).to_numpy()
        _, codes = np.unique(codes, return_inverse=True)
        return codes, True, int(codes.max()) + 1
    labels = s.astype(object).where(s.notna(), "<NA>").astype(str)
    codes, uniques = pd.factorize(labels)
    if len(uniques) > max_card:
        return None
    return codes, False, len(uniques)


def sensitive_diversity(df, qi, sens_cols, l_target=2, t_target=0.3):
    """
    Для каждой чувствительной колонки:
      l  — число различных значений в классе эквивалентности (distinct l-diversity);
      t  — расстояние распределения класса от общего: EMD для порядковых,
           полное отклонение (TVD) для категориальных.
    """
    qi = [c for c in qi if c in df.columns]
    rows = []
    if not qi or df.empty:
        return pd.DataFrame(rows)
    codes = equivalence_codes(df, qi)
    n_classes = int(codes.max()) + 1
    sizes = np.bincount(codes, minlength=n_classes).astype(float)
    for col in sens_cols:
        if col not in df.columns or col in qi:
            continue
        enc = _encode_sensitive(df[col])
        if enc is None or n_classes * enc[2] > 50_000_000:
            rows.append({"Колонка": col, "Тип": "пропущена (слишком много значений)"})
            continue
        sc, ordered, m = enc
        mat = np.bincount(codes * m + sc, minlength=n_classes * m).reshape(n_classes, m).astype(float)
        distinct = (mat > 0).sum(axis=1)
        p = mat / sizes[:, None]
        q = mat.sum(axis=0) / mat.sum()
        if ordered and m > 1:
            dist = np.abs(np.cumsum(p - q, axis=1)).sum(axis=1) / (m - 1)
        else:
            dist = 0.5 * np.abs(p - q).sum(axis=1)
        rows.append({
            "Колонка": col,
            "Тип": "порядковая (EMD)" if ordered else "категориальная (TVD)",
            "Мин. l": int(distinct.min()),
            "Записей в классах с l < L (%)": round(float((distinct[codes] < l_target).mean()) * 100, 2),
            "Макс. t": round(float(dist.max()), 3),
            "Записей в классах с t > T (%)": round(float((dist[codes] > t_target).mean()) * 100, 2),
        })
    return pd.DataFrame(rows)


# ---------- атака связывания ----------
def attacker_view(df_raw, qi, bin_steps, cat_maps):
    """Исходные QI так, как их видит атакующий после тех же обобщений."""
    view = df_raw[qi].copy()
    for c in qi:
        if c in cat_maps:
            view[c] = view[c].map(cat_maps[c]).where(view[c].notna(), view[c])
        if c in bin_steps:
            view[c] = generalize_numeric_series(view[c], bin_steps[c])
    return view


def linkage_attack(df_raw, df_safe, qi, bin_steps, cat_maps, pos_col="_orig_pos"):
    """
    Атакующий знает исходные QI человека и ищет подходящие строки в выгрузке.
    Истинное соответствие известно (pos_col), поэтому можно измерить успех.
    Замаскированные строки («*») считаются подходящими для всех.
    """
    qi = [c for c in qi if c in df_safe.columns]
    if not qi or len(df_safe) == 0:
        return None
    n_raw, n_safe = len(df_raw), len(df_safe)
    ov = attacker_view(df_raw, qi, bin_steps, cat_maps).reset_index(drop=True)
    sv = df_safe[qi].reset_index(drop=True)
    both = pd.concat([sv.astype(str), ov.astype(str)], ignore_index=True).fillna("<NA>")
    codes = both.groupby(qi, sort=False, dropna=False).ngroup().to_numpy()
    safe_codes, orig_codes = codes[:n_safe], codes[n_safe:]
    counts = np.bincount(safe_codes, minlength=int(codes.max()) + 1)
    masked = (sv.astype(str) == "*").all(axis=1).to_numpy()
    masked_n = int(masked.sum())

    pos = df_safe[pos_col].to_numpy()
    row_of = np.full(n_raw, -1)
    row_of[pos] = np.arange(n_safe)
    pop = np.flatnonzero(row_of >= 0)
    if len(pop) == 0:
        return None
    j = row_of[pop]
    cands = counts[orig_codes[pop]] + masked_n
    true_in = (safe_codes[j] == orig_codes[pop]) | masked[j]
    return {
        "population": int(len(pop)),
        "dropped_share": round((1 - len(pop) / n_raw) * 100, 2),
        "avg_candidates": round(float(cands.mean()), 2),
        "unique_share": round(float((cands == 1).mean()) * 100, 2),
        "correct_unique_share": round(float(((cands == 1) & true_in).mean()) * 100, 2),
        "expected_reid": round(float((true_in / np.maximum(cands, 1)).mean()) * 100, 2),
    }


# ---------- полезность ----------
def ks_statistic(a, b):
    """Двухвыборочная статистика Колмогорова–Смирнова (0 — одинаковые распределения)."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side="right") / len(a)
    cdf_b = np.searchsorted(b, allv, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _tvd(o, s):
    po, ps = o.value_counts(normalize=True), s.value_counts(normalize=True)
    idx = po.index.union(ps.index)
    return 0.5 * float(np.abs(po.reindex(idx, fill_value=0) - ps.reindex(idx, fill_value=0)).sum())


def utility_report(df_raw, df_safe, cols):
    """Сравнение распределений до/после. Возвращает (таблица, Δкорреляций | None)."""
    rows, num_cols = [], []
    for c in cols:
        if c not in df_raw.columns or c not in df_safe.columns:
            continue
        o, s = df_raw[c], df_safe[c]
        if pd.api.types.is_numeric_dtype(o) and not pd.api.types.is_bool_dtype(o):
            ov, sv = o.astype(float).dropna(), to_numeric_view(s).dropna()
            if len(ov) == 0 or len(sv) == 0:
                rows.append({"Колонка": c, "Тип": "числовая", "Метрика": "KS", "Значение": np.nan,
                             "Δ среднего (%)": np.nan, "Δ std (%)": np.nan})
                continue
            num_cols.append(c)
            rows.append({
                "Колонка": c, "Тип": "числовая", "Метрика": "KS",
                "Значение": round(ks_statistic(ov, sv), 3),
                "Δ среднего (%)": round(abs(sv.mean() - ov.mean()) / (abs(ov.mean()) + 1e-9) * 100, 2),
                "Δ std (%)": round(abs(sv.std() - ov.std()) / (abs(ov.std()) + 1e-9) * 100, 2)
                if len(ov) > 1 and len(sv) > 1 else np.nan,
            })
        else:
            rows.append({"Колонка": c, "Тип": "категориальная", "Метрика": "TVD",
                         "Значение": round(_tvd(o.dropna().astype(str), s.dropna().astype(str)), 3),
                         "Δ среднего (%)": np.nan, "Δ std (%)": np.nan})
    corr_diff = None
    if len(num_cols) >= 2:
        co = df_raw[num_cols].astype(float).corr().to_numpy()
        cs = pd.DataFrame({c: to_numeric_view(df_safe[c]) for c in num_cols}).corr().to_numpy()
        diff = np.abs(co - cs)[np.triu_indices(len(num_cols), 1)]
        diff = diff[np.isfinite(diff)]
        corr_diff = round(float(diff.mean()), 3) if diff.size else None
    return pd.DataFrame(rows), corr_diff


# ---------- смещение из-за удаления строк ----------
def suppression_bias(df_raw, qi, kept_pos):
    """
    Как удаление строк (suppression) исказило состав выборки по каждому QI:
    TVD между распределением до/после и категория, из которой удалено больше всего.
    """
    qi = [c for c in qi if c in df_raw.columns]
    n = len(df_raw)
    kept = np.zeros(n, dtype=bool)
    kept[np.asarray(kept_pos, dtype=int)] = True
    rows = []
    for c in qi:
        col = df_raw[c]
        if pd.api.types.is_numeric_dtype(col) and col.nunique() > 8:
            try:
                col = pd.qcut(col, 5, duplicates="drop")
            except ValueError:
                pass
        s = col.astype(str).reset_index(drop=True)
        tot = s.value_counts()
        kept_c = s[kept].value_counts().reindex(tot.index, fill_value=0)
        removed_rate = (1 - kept_c / tot) * 100
        p0, p1 = tot / tot.sum(), kept_c / max(kept_c.sum(), 1)
        rows.append({"Колонка": c, "TVD до/после": round(float(0.5 * (p0 - p1).abs().sum()), 4),
                     "Больше всего удалено из": str(removed_rate.idxmax()),
                     "Удалено из неё (%)": round(float(removed_rate.max()), 1)})
    return pd.DataFrame(rows)


# ---------- агрегаты для графиков (исходная таблица после запуска не нужна) ----------
def distribution_summary(df_raw, df_safe, cols, bins=20, top=20):
    """
    Сводные распределения «до/после» по колонкам: числа — гистограммы, категории — частоты top-N.
    Это агрегаты, а не строки; позволяют строить графики после удаления исходного массива.
    """
    out = {}
    for c in cols:
        if c not in df_raw.columns or c not in df_safe.columns:
            continue
        o, s = df_raw[c], df_safe[c]
        if pd.api.types.is_numeric_dtype(o) and not pd.api.types.is_bool_dtype(o):
            ov, sv = o.astype(float).dropna(), to_numeric_view(s).dropna()
            if ov.empty:
                continue
            lo = float(min(ov.min(), sv.min() if len(sv) else ov.min()))
            hi = float(max(ov.max(), sv.max() if len(sv) else ov.max()))
            if hi <= lo:
                hi = lo + 1.0
            edges = np.linspace(lo, hi, bins + 1)
            hb = np.histogram(ov, edges)[0] / max(len(ov), 1)
            ha = np.histogram(sv, edges)[0] / max(len(sv), 1) if len(sv) else np.zeros(bins)
            out[c] = {"type": "numeric", "centers": ((edges[:-1] + edges[1:]) / 2).round(4).tolist(),
                      "before": hb.round(5).tolist(), "after": ha.round(5).tolist()}
        else:
            po = o.dropna().astype(str).value_counts(normalize=True).head(top)
            ps = s.dropna().astype(str).value_counts(normalize=True)
            out[c] = {"type": "categorical", "labels": list(po.index),
                      "before": po.round(5).tolist(), "after": [round(float(ps.get(i, 0)), 5) for i in po.index]}
    return out

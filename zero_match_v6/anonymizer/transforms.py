# -*- coding: utf-8 -*-
"""Преобразования: перестановка, шум, биннинг, категории, цепи Маркова."""
import json
from collections import defaultdict

import numpy as np
import pandas as pd


# ---------- перестановка ----------
def sattolo_derangement(items, rng):
    """
    Алгоритм Саттоло: перестановка без неподвижных точек (один цикл длины n).
    Это лишь подмножество всех беспорядков (субфакториала !n).
    """
    result = list(items)
    for i in range(len(result) - 1, 0, -1):
        j = rng.randrange(i)
        result[i], result[j] = result[j], result[i]
    return result


# ---------- шум ----------
def is_noisable(series):
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _finish(noisy, series):
    m = series.min()
    if pd.notna(m) and m >= 0:            # пост-обработка: не нарушает ДП
        noisy = noisy.clip(lower=0.0)
    if pd.api.types.is_integer_dtype(series):
        return noisy.round().astype(series.dtype)
    return noisy.round(2)


def add_noise_pct(series, noise_pct, np_rng):
    """Эвристика: шкала Лапласа ∝ значению (min 1.0). НЕ даёт гарантий ДП."""
    if noise_pct <= 0 or not is_noisable(series):
        return series
    x = series.astype(float)
    scale = np.nan_to_num(np.maximum(x.abs().to_numpy() * noise_pct / 100.0, 1.0), nan=1.0)
    return _finish(x + np_rng.laplace(0.0, scale), series)


def add_noise_ldp(series, epsilon, np_rng, clip_pct=1.0):
    """
    Локальная дифференциальная приватность по значению: значение обрезается до
    [lo, hi], шум Лапласа со шкалой (hi - lo) / ε  =>  ε-LDP для этой колонки.
    Границы берутся из процентилей данных — строго говоря, это утечка;
    для формальных гарантий задавайте границы вручную из предметной области.
    Возвращает (series, info | None).
    """
    if not is_noisable(series) or epsilon <= 0:
        return series, None
    x = series.astype(float)
    valid = x.dropna()
    if valid.empty:
        return series, None
    lo, hi = np.percentile(valid, [clip_pct, 100 - clip_pct])
    if hi <= lo:
        hi = lo + 1.0
    scale = (hi - lo) / epsilon
    noisy = x.clip(lo, hi) + np_rng.laplace(0.0, scale, size=len(x))
    info = {"epsilon": float(epsilon), "lo": float(lo), "hi": float(hi), "scale": float(scale)}
    return _finish(noisy, series), info


# ---------- числовое обобщение ----------
def auto_step(series):
    try:
        numeric = pd.to_numeric(series, errors='coerce').dropna()
        if numeric.empty:
            return 10
        mx = float(numeric.abs().max())
        if mx > 10_000:
            return 5_000
        if mx > 1_000:
            return 500
        if mx > 100:
            return 50
        return 10
    except Exception:
        return 10


def generalize_numeric_series(series, step=10):
    """Векторный биннинг: 37 -> '30-39'. Нечисловые значения остаются как есть."""
    x = pd.to_numeric(series, errors="coerce")
    lower = np.floor(x / step) * step
    out = lower.astype("Int64").astype(str) + "-" + (lower + step - 1).astype("Int64").astype(str)
    return out.where(x.notna(), series)


_NUM = r'(-?\d+(?:[.,]\d+)?)'
_PAT_DASH = rf'^\s*{_NUM}\s*[-–]\s*{_NUM}\s*$'
_PAT_BRACKET = rf'^\s*[\[\]\(\)]*\s*{_NUM}\s*[,;]\s*{_NUM}\s*[\[\]\(\)]*\s*$'


def interval_midpoints(series):
    """Середины интервалов: 20-29, -20--10, [20, 30[, (20; 30). Прочее -> NaN."""
    ser = series.astype(object).map(lambda v: v if isinstance(v, str) else None)
    if not ser.notna().any():
        return pd.Series(np.nan, index=series.index, dtype=float)
    e = ser.str.extract(_PAT_DASH).fillna(ser.str.extract(_PAT_BRACKET))

    def num(col):
        return pd.to_numeric(col.astype("string").str.replace(",", ".", regex=False),
                             errors="coerce")
    return (num(e[0]) + num(e[1])) / 2


def to_numeric_view(series):
    """Число для анализа: числа как есть, интервалы -> середины."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    return interval_midpoints(series).fillna(pd.to_numeric(series, errors="coerce"))


def info_loss_numeric(original_series, anonymized_series):
    try:
        orig = pd.to_numeric(original_series, errors='coerce').dropna()
        if orig.empty:
            return 0.0
        anon = to_numeric_view(anonymized_series).dropna()
        if anon.empty:
            return 100.0
        om, am = float(orig.mean()), float(anon.mean())
        os_ = float(orig.std()) if len(orig) > 1 else 0.0
        as_ = float(anon.std()) if len(anon) > 1 else 0.0
        mean_loss = abs(om - am) / (abs(om) + 1e-9)
        std_loss = abs(os_ - as_) / (abs(os_) + 1e-9) if os_ > 0 else 0.0
        return round((mean_loss + std_loss) / 2 * 100, 2)
    except Exception:
        return 0.0


# ---------- категории ----------
def parse_hierarchy(text):
    """JSON вида {"Колонка": {"значение": "обобщение"}}. Пустая строка -> {}."""
    if not text or not text.strip():
        return {}
    data = json.loads(text)
    if not isinstance(data, dict) or not all(isinstance(v, dict) for v in data.values()):
        raise ValueError('Ожидается JSON вида {"Колонка": {"значение": "обобщение"}}')
    return data


def generalize_categories(series, hierarchy=None, min_count=0, other_label="Другое"):
    """
    1) значения заменяются по иерархии (город -> регион);
    2) значения реже min_count объединяются в «Другое».
    Возвращает (series, final_map) — отображение «исходное -> итоговое».
    """
    hierarchy = hierarchy or {}
    uniques = series.dropna().unique().tolist()
    step1 = {u: hierarchy.get(u, u) for u in uniques}
    mapped = series.map(step1)
    final_map = dict(step1)
    if min_count and min_count > 0:
        counts = mapped.value_counts()
        rare = set(counts[counts < min_count].index)
        if rare:
            final_map = {u: (other_label if v in rare else v) for u, v in step1.items()}
    out = series.map(final_map)
    return out.where(series.notna(), series), final_map


# ---------- цепи Маркова ----------
class MarkovBehaviorGenerator:
    def __init__(self):
        self.transitions = defaultdict(list)

    def fit(self, sequences):
        for seq in sequences:
            if len(seq) < 2:
                continue
            for i in range(len(seq) - 1):
                self.transitions[seq[i]].append(seq[i + 1])
            self.transitions[seq[-1]].append(None)

    def generate(self, start_state, rng, max_steps=5):
        if start_state not in self.transitions:
            return [start_state]
        current, chain = start_state, [start_state]
        for _ in range(max_steps - 1):
            possible = self.transitions.get(current, [])
            if not possible:
                break
            nxt = rng.choice(possible)
            if nxt is None:
                break
            chain.append(nxt)
            current = nxt
        return chain

    def safe_generate(self, starts, rng, max_steps=5, min_len=2):
        for _ in range(10):
            chain = self.generate(rng.choice(starts), rng, max_steps)
            if len(chain) >= min_len:
                return chain
        chain = self.generate(rng.choice(starts), rng, max_steps)
        while len(chain) < min_len:
            chain.append("(конец)")
        return chain

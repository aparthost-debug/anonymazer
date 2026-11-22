# -*- coding: utf-8 -*-
"""
Единый источник правды по ПДн: детекторы, классификация колонок, рекомендации.

Уровни уверенности находок:
  high   — проверена контрольная сумма / есть контекст (блокируют экспорт);
  medium — шаблон без проверки (ФИО, адрес) — показываем, не блокируем;
  low    — «похоже на номер» (10/12 цифр без контрольной суммы) — только предупреждение.
"""
import re

import numpy as np
import pandas as pd

from .text_clean import (ADDRESS_RE, BDATE_CTX_RE, CARD_RE, EMAIL_RE, IBAN_RE, INN_CTX_RE,
                         INN_RE, INTL_PHONE_RE, NAME_FULL_RE, PASSPORT_CTX_RE,
                         PASSPORT_SERIES_RE, PHONE_RE, SNILS_RE, card_ok, iban_ok, inn_ok,
                         snils_ok)

COL_FULL, COL_PART = "Полное совпадение (%)", "Вхождение в тексте (%)"
_digits = lambda s: re.sub(r'\D', '', s)  # noqa: E731


def _full(rx, chk=None, raw=False):
    def f(s):
        s = s.strip()
        if not rx.fullmatch(s):
            return False
        return chk(s if raw else _digits(s)) if chk else True
    return f


def _part(rx, chk=None, raw=False):
    def f(s):
        for m in rx.finditer(s):
            g = m.group(0)
            if chk is None or chk(g if raw else _digits(g)):
                return True
        return False
    return f


def _any(*fs):
    return lambda s: any(f(s) for f in fs)


_never = lambda s: False  # noqa: E731

# имя типа -> (полное совпадение, вхождение, уверенность)
PII_TYPES = {
    "email": (_full(EMAIL_RE), _part(EMAIL_RE), "high"),
    "телефон": (_any(_full(PHONE_RE), _full(INTL_PHONE_RE)),
                _any(_part(PHONE_RE), _part(INTL_PHONE_RE)), "high"),
    "ИНН": (_full(INN_RE, inn_ok), _any(_part(INN_RE, inn_ok), _part(INN_CTX_RE)), "high"),
    "СНИЛС": (_full(SNILS_RE, snils_ok), _part(SNILS_RE, snils_ok), "high"),
    "банковская карта": (_full(CARD_RE, card_ok), _part(CARD_RE, card_ok), "high"),
    "IBAN": (_full(IBAN_RE, iban_ok, raw=True), _part(IBAN_RE, iban_ok, raw=True), "high"),
    "паспорт (с контекстом)": (_never, lambda s: bool(PASSPORT_CTX_RE.search(s)
                                                       or PASSPORT_SERIES_RE.search(s)), "high"),
    "дата рождения": (_never, _part(BDATE_CTX_RE), "high"),
    "ФИО": (_full(NAME_FULL_RE), _never, "medium"),
    "адрес": (_never, _part(ADDRESS_RE), "medium"),
    "номер 10/12 цифр (без проверки суммы)": (_full(INN_RE), _part(INN_RE), "low"),
}


def detect_pii(df, sample_size=2000):
    """Таблица: колонка, тип, уверенность, доля полных совпадений и вхождений (%)."""
    rows = []
    for c in df.columns:
        s = df[c].dropna()
        if s.empty:
            continue
        if len(s) > sample_size:
            s = s.sample(sample_size, random_state=0)
        vals = s.astype(str).tolist()
        for name, (full, part, conf) in PII_TYPES.items():
            f = float(np.mean([full(v) for v in vals])) * 100
            p = float(np.mean([part(v) for v in vals])) * 100
            if f > 0 or p > 0:
                rows.append({"Колонка": c, "Тип ПДн": name, "Уверенность": conf,
                             COL_FULL: round(f, 1), COL_PART: round(p, 1)})
    return pd.DataFrame(rows, columns=["Колонка", "Тип ПДн", "Уверенность", COL_FULL, COL_PART])


def count_pii_cells(df, max_unique=50_000, exclude=("ФИО",)):
    """
    Контроль результата. total/by_column — только high (блокируют экспорт);
    low_total/low_by_column — подозрительные числа без контрольной суммы (предупреждение).
    «ФИО» исключено: шаблон имени ложно срабатывает на «Нижний Новгород».
    """
    high_t = [(f, p) for n, (f, p, c) in PII_TYPES.items() if c == "high" and n not in exclude]
    other_t = [(f, p) for n, (f, p, c) in PII_TYPES.items() if c in ("low", "medium")
               and n not in exclude and not n.startswith("адрес")]
    by_col, low_by_col, total, low_total = {}, {}, 0, 0
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            continue
        vc = s.dropna().astype(str).value_counts()
        if vc.empty:
            continue
        if len(vc) > max_unique:
            vc = vc.head(max_unique)
        n_hi = n_lo = 0
        for v, cnt in vc.items():
            if any(f(v) or p(v) for f, p in high_t):
                n_hi += int(cnt)
            elif any(f(v) or p(v) for f, p in other_t):
                n_lo += int(cnt)
        if n_hi:
            by_col[c], total = n_hi, total + n_hi
        if n_lo:
            low_by_col[c], low_total = n_lo, low_total + n_lo
    return {"total": total, "by_column": by_col, "low_total": low_total,
            "low_by_column": low_by_col}


# ---------- классификация колонок по названию (по токенам, не по подстроке) ----------
def _tokens(name):
    s = re.sub(r'([a-zа-яё])([A-ZА-ЯЁ])', r'\1 \2', str(name))
    return [t for t in re.split(r'[^0-9A-Za-zА-Яа-яЁё]+', s.lower()) if t]


def _match(tokens, exact, prefixes=()):
    return any(t in exact or any(t.startswith(p) for p in prefixes) for t in tokens)


_DIRECT_HIGH = ({"фио", "инн", "снилс", "имя", "email", "mail", "phone", "tel", "uuid", "iban",
                 "bic", "swift", "name", "fio", "телефон", "паспорт", "passport"},
                ("фамил", "отчеств", "телеф", "мобил", "почт", "адрес", "улиц", "квартир",
                 "phone", "email", "address", "surname", "firstname", "lastname", "fullname"))
_DIRECT_MED = ({"клиент", "client", "customer", "user", "account", "аккаунт", "id"}, ())
_SENSITIVE = ({"sum", "amount"},
              ("зарплат", "доход", "income", "salary", "покуп", "сумм", "расход", "expense",
               "здоров", "медицин", "диагноз", "health", "medical", "кредит", "долг", "debt",
               "финанс", "financial", "истори", "religi", "полит", "биометр", "credit"))
_QI = ({"age", "пол", "sex", "gender", "zip", "city", "region", "country", "возраст", "город",
        "регион", "страна", "индекс", "postcode"},
       ("рожд", "birth", "должност", "профес", "образован", "семейн", "стаж", "occupation",
        "marital"))


def classify_columns(df):
    """Эвристические роли колонок. Это подсказки, а не юридическая классификация."""
    rows = []
    for col in df.columns:
        s, tk = df[col], _tokens(col)
        non_null = int(s.notna().sum())
        ratio = (float(s.nunique(dropna=True)) / non_null) if non_null else 0.0
        is_float = pd.api.types.is_float_dtype(s)
        is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        by_sens, by_qi = _match(tk, *_SENSITIVE), _match(tk, *_QI)
        if _match(tk, *_DIRECT_HIGH):
            role, conf, basis, why = ("direct_identifier", "high", "name",
                                      "Название колонки указывает на идентификатор")
        elif _match(tk, *_DIRECT_MED):
            role, conf, basis, why = ("direct_identifier", "medium", "name",
                                      "Название похоже на идентификатор клиента/аккаунта")
        elif non_null >= 20 and ratio >= 0.98 and not is_float and not (by_sens or by_qi):
            role, conf, basis, why = ("direct_identifier", "medium", "unique",
                                      "Почти все значения уникальны — возможно, идентификатор")
        elif by_sens:
            role, conf, basis, why = ("sensitive", "medium", "name", "Название похоже на чувствительный атрибут")
        elif by_qi or is_num or pd.api.types.is_datetime64_any_dtype(s):
            role, conf, basis, why = ("quasi_identifier", "medium", "name" if by_qi else "dtype",
                                      "Потенциальный квазиидентификатор по имени или типу")
        else:
            role, conf, basis, why = ("ordinary", "low", "-", "Явных признаков не найдено")
        rows.append({"column": str(col), "dtype": str(s.dtype), "suggested_role": role,
                     "confidence": conf, "basis": basis, "reason": why,
                     "missing_pct": round(float(s.isna().mean() * 100), 2),
                     "unique_count": int(s.nunique(dropna=True))})
    return pd.DataFrame(rows)


def unhandled_direct_identifiers(df, handled):
    """Колонки, которые ПО НАЗВАНИЮ — прямые идентификаторы (high), но ни в одном шаге."""
    cl = classify_columns(df)
    mask = (cl["suggested_role"] == "direct_identifier") & (cl["confidence"] == "high")
    return [c for c in cl.loc[mask, "column"] if c not in set(handled)]


def suggest_config(df, pii_df=None, classification=None):
    """Рекомендованные списки колонок (только существующие в df)."""
    pii_df = detect_pii(df) if pii_df is None else pii_df
    cl = classify_columns(df) if classification is None else classification
    cols = df.columns.tolist()
    role = cl.set_index("column")
    is_num = {c: pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
              for c in cols}

    id_cols, text_cols = [], []
    for c in cols:
        sub = pii_df[(pii_df["Колонка"] == c) & (pii_df["Уверенность"] != "low")]
        full = float(sub[COL_FULL].max()) if not sub.empty else 0.0
        part = float(sub[COL_PART].max()) if not sub.empty else 0.0
        r = role.loc[str(c)]
        if full >= 50 or (r["suggested_role"] == "direct_identifier" and r["basis"] == "name"):
            id_cols.append(c)
        elif part >= 2 and not is_num[c]:
            text_cols.append(c)

    rest = [c for c in cols if c not in id_cols and c not in text_cols]
    sens = [c for c in rest if role.loc[str(c), "suggested_role"] == "sensitive"]
    derange = [c for c in sens if is_num[c]]
    qi = [c for c in rest if role.loc[str(c), "suggested_role"] == "quasi_identifier"
          and role.loc[str(c), "basis"] == "name"]
    kanon = [c for c in qi if is_num[c]]
    cat = [c for c in qi if not is_num[c]]

    markov = None
    for c in rest:
        if is_num[c]:
            continue
        vals = df[c].dropna().astype(str)
        if len(vals) and vals.head(500).str.contains(' -> ', regex=False).mean() >= 0.5:
            markov = c
            break

    return {"id_cols": id_cols, "text_cols": text_cols, "derange_cols": derange,
            "kanon_cols": kanon, "quasi_cols": qi, "cat_cols": cat,
            "sensitive_cols": sens, "markov_col": markov}

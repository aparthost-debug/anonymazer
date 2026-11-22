# -*- coding: utf-8 -*-
"""Зачистка свободного текста: контакты, документы, адреса, имена (+ NER)."""
import re
from functools import lru_cache

import pandas as pd

EMAIL_RE = re.compile(r'[\w.\-]+@[\w\-]+(?:\.[\w\-]+)+')
PHONE_RE = re.compile(
    r'(?<![\w+])(?:\+7|7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)')
INTL_PHONE_RE = re.compile(r'(?<!\w)\+\d(?:[\s().-]*\d){7,14}(?!\w)')
CARD_RE = re.compile(r'(?<!\d)(?:\d[ \-]?){12,18}\d(?!\d)')
SNILS_RE = re.compile(r'(?<!\d)\d{3}[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{2}(?!\d)')
INN_RE = re.compile(r'(?<!\d)(?:\d{12}|\d{10})(?!\d)')
INN_CTX_RE = re.compile(r'(?i)(\bинн\s*[:№#]?\s*)(\d{12}|\d{10})(?!\d)')
IBAN_RE = re.compile(
    r'(?<![A-Za-z0-9])[A-Za-z]{2}\d{2}(?: ?[A-Za-z0-9]{4}){2,7}(?: ?[A-Za-z0-9]{1,3})?(?![A-Za-z0-9])')
PASSPORT_CTX_RE = re.compile(r'(?i)(паспорт\w*[^\d\n]{0,20})(\d{2}\s?\d{2}\s?\d{6})(?!\d)')
PASSPORT_SERIES_RE = re.compile(r'(?i)(сери[яи]\s*)(\d{2}\s?\d{2})(\s*(?:номер|№|N)?\s*)(\d{6})(?!\d)')
ADDRESS_RE = re.compile(
    r'(?i)\b(?:ул\.?|улица|пр-т|проспект|пер\.?|переулок|б-р|бульвар|шоссе)\s+'
    r'[А-ЯЁA-Z][\w\-]*(?:[,\s]+(?:д\.?|дом)?\s*\d+[\w/]*)?(?:[,\s]+(?:кв\.?|квартира)\s*\d+)?')
BDATE_CTX_RE = re.compile(r'(?i)((?:родил\w*|рожд\w*|д\.?\s?р\.?)[^\d\n]{0,15})(\d{2}[./]\d{2}[./]\d{2,4})')
_W = r'[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?'
NAME_FULL_RE = re.compile(rf'^{_W}(?:\s{_W}){{1,2}}$')   # «Санкт-Петербург» — не ФИО


# ---------- контрольные суммы ----------
def luhn_ok(digits):
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def inn_ok(digits):
    d = [int(x) for x in digits]
    if len(d) == 10:
        w = [2, 4, 10, 3, 5, 9, 4, 6, 8]
        return (sum(a * b for a, b in zip(d, w)) % 11) % 10 == d[9]
    if len(d) == 12:
        w11 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        w12 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        n11 = (sum(a * b for a, b in zip(d[:10], w11)) % 11) % 10
        n12 = (sum(a * b for a, b in zip(d[:11], w12)) % 11) % 10
        return n11 == d[10] and n12 == d[11]
    return False


def snils_ok(digits):
    if len(digits) != 11:
        return False
    d = [int(x) for x in digits]
    s = sum(d[i] * (9 - i) for i in range(9))
    ctrl = s if s < 100 else (0 if s in (100, 101) else s % 101)
    if ctrl == 100:
        ctrl = 0
    return ctrl == int(digits[9:])


def card_ok(digits):
    return 13 <= len(digits) <= 19 and luhn_ok(digits)


def iban_ok(raw):
    s = re.sub(r'\s', '', raw).upper()
    if not 15 <= len(s) <= 34 or not s.isalnum():
        return False
    moved = s[4:] + s[:4]
    return int(''.join(str(int(c, 36)) for c in moved)) % 97 == 1


# ---------- имена ----------
def _name_piece(name):
    """Слово из ФИО -> шаблон с учётом падежных окончаний."""
    if re.search(r'[\s_\d]', name) or len(name) < 4:
        return re.escape(name)
    stem = name[:-1] if name[-1].lower() in 'аяйьеоы' else name
    return re.escape(stem) + r'\w{0,3}'


@lru_cache(maxsize=8192)
def _names_regex(names):
    return re.compile(r'(?<!\w)(?:' + '|'.join(_name_piece(n) for n in names) + r')(?!\w)',
                      re.IGNORECASE)


def redact_names(text, names):
    if not names or not isinstance(text, str):
        return text
    return _names_regex(tuple(names)).sub('[ИМЯ УДАЛЕНО]', text)


def collect_names(df, id_cols):
    """Для каждой строки: значения ID-колонок и их слова (≥3 символов), длинные первыми."""
    per_row = [set() for _ in range(len(df))]
    for col in id_cols:
        for i, v in enumerate(df[col].tolist()):
            if isinstance(v, str) and v.strip():
                per_row[i].add(v.strip())
                per_row[i].update(w for w in re.split(r'[\s,;]+', v) if len(w) >= 3)
    return [tuple(sorted(s, key=len, reverse=True)) for s in per_row]


# ---------- NER: spaCy (локальная модель) -> natasha ----------
PERSON_LABELS = {"PER", "PERSON"}


@lru_cache(maxsize=4)
def _spacy_model(name):
    """Только уже установленная модель, без загрузки из сети."""
    if not name:
        return None
    try:
        import spacy
        return spacy.load(name)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _natasha():
    try:
        from natasha import Doc, NewsEmbedding, NewsNERTagger, Segmenter
        return Doc, Segmenter(), NewsNERTagger(NewsEmbedding())
    except Exception:
        return None


def ner_backend(model=""):
    """'spacy:<модель>' | 'natasha' | None — что реально доступно."""
    if _spacy_model(model) is not None:
        return f"spacy:{model}"
    return "natasha" if _natasha() is not None else None


@lru_cache(maxsize=4)
def get_ner(model=""):
    """Функция text -> text с вырезанием ТОЛЬКО имён людей (города и т.п. остаются)."""
    nlp = _spacy_model(model)
    if nlp is not None:
        def ner_spacy(text):
            try:
                spans = [(e.start_char, e.end_char) for e in nlp(text[:100000]).ents
                         if e.label_ in PERSON_LABELS]
            except Exception:
                return text
            for a, b in sorted(spans, reverse=True):
                text = text[:a] + '[ИМЯ УДАЛЕНО]' + text[b:]
            return text
        return ner_spacy
    nat = _natasha()
    if nat is None:
        return None
    Doc, segmenter, tagger = nat

    def ner_natasha(text):
        try:
            doc = Doc(text)
            doc.segment(segmenter)
            doc.tag_ner(tagger)
            for span in sorted(doc.spans, key=lambda s: s.start, reverse=True):
                if span.type == "PER":
                    text = text[:span.start] + '[ИМЯ УДАЛЕНО]' + text[span.stop:]
        except Exception:
            pass
        return text
    return ner_natasha


# ---------- очистка ----------
def _sub_valid(regex, text, check, label):
    def repl(m):
        return label if check(re.sub(r'\D', '', m.group(0)) if label != '[IBAN УДАЛЕН]' else m.group(0)) \
            else m.group(0)
    return regex.sub(repl, text)


def clean_free_text(text, names=(), ner=None):
    """Удаляет e-mail, IBAN, карты, телефоны, СНИЛС, ИНН, паспорта, адреса, даты рождения, имена."""
    if not isinstance(text, str):
        return text
    t = EMAIL_RE.sub('[EMAIL УДАЛЕН]', text)
    t = _sub_valid(IBAN_RE, t, iban_ok, '[IBAN УДАЛЕН]')
    t = _sub_valid(CARD_RE, t, card_ok, '[КАРТА УДАЛЕНА]')
    t = PHONE_RE.sub('[ТЕЛЕФОН УДАЛЕН]', t)
    t = INTL_PHONE_RE.sub('[ТЕЛЕФОН УДАЛЕН]', t)
    t = _sub_valid(SNILS_RE, t, snils_ok, '[СНИЛС УДАЛЕН]')
    t = INN_CTX_RE.sub(r'\1[ИНН УДАЛЕН]', t)           # «ИНН 123…» — без проверки суммы
    t = _sub_valid(INN_RE, t, inn_ok, '[ИНН УДАЛЕН]')
    t = PASSPORT_CTX_RE.sub(r'\1[ПАСПОРТ УДАЛЕН]', t)
    t = PASSPORT_SERIES_RE.sub(r'\1[ПАСПОРТ УДАЛЕН]', t)
    t = ADDRESS_RE.sub('[АДРЕС УДАЛЕН]', t)
    t = BDATE_CTX_RE.sub(r'\1[ДАТА УДАЛЕНА]', t)
    if ner:
        t = ner(t)
    return redact_names(t, names)


def clean_series(series, names_per_row=None, ner_model=None):
    """
    Регулярки считаются один раз на уникальное значение, имена (зависят от строки)
    применяются поверх. ner_model=None — NER выключен; "" — только natasha.
    """
    ner = get_ner(ner_model) if ner_model is not None else None
    cache, out = {}, []
    for i, v in enumerate(series.tolist()):
        if not isinstance(v, str):
            out.append(v)
            continue
        base = cache.get(v)
        if base is None:
            base = cache[v] = clean_free_text(v, (), ner)
        if names_per_row and names_per_row[i]:
            base = redact_names(base, names_per_row[i])
        out.append(base)
    return pd.Series(out, index=series.index)

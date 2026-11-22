# -*- coding: utf-8 -*-
"""Конфигурация конвейера и её сохранение/загрузка (JSON)."""
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Optional

NONE_OPTION = "Не использовать"


@dataclass
class Config:
    # --- колонки ---
    id_cols: list = field(default_factory=list)          # токенизация
    derange_cols: list = field(default_factory=list)     # перестановка + шум
    kanon_cols: list = field(default_factory=list)       # биннинг чисел
    cat_cols: list = field(default_factory=list)         # обобщение категорий
    text_cols: list = field(default_factory=list)        # зачистка текста
    quasi_cols: list = field(default_factory=list)       # QI для оценки риска
    sensitive_cols: list = field(default_factory=list)   # для l-разнообразия / t-близости
    markov_col: str = NONE_OPTION
    hierarchy_json: str = ""                             # {"Город": {"Тверь": "Центр"}}
    manual_steps_json: str = ""                          # {"Возраст": 10} — приоритетнее автоподбора
    # --- шум ---
    noise_mode: str = "pct"                              # "pct" | "ldp"
    noise_pct: int = 5
    epsilon: float = 5.0
    dp_clip_pct: float = 1.0
    # --- цели приватности ---
    k_target: int = 5
    l_target: int = 2
    t_target: float = 0.3
    enforce_l: bool = True                               # нарушение l блокирует экспорт
    enforce_t: bool = False                              # t строгая: по умолчанию только предупреждение
    # --- обобщение и подавление ---
    auto_step: bool = True
    supp_mode: str = "mask"                              # "none" | "mask" | "drop"
    supp_limit: float = 5.0                              # допустимая доля подавления (%)
    rare_min_count: int = 5
    use_ner: bool = False
    ner_model: str = "ru_core_news_lg"                   # spaCy; если нет — natasha
    # --- прочее ---
    seed: Optional[int] = None
    use_arx: bool = False
    arx_jar: str = "arx-cli.jar"
    arx_k: int = 5
    arx_l: int = 2


COLUMN_LIST_FIELDS = ("id_cols", "derange_cols", "kanon_cols", "cat_cols",
                      "text_cols", "quasi_cols", "sensitive_cols")

_SKIP = object()


def _coerce(default, value):
    """Приводит значение из JSON к типу поля или возвращает _SKIP."""
    if isinstance(default, bool):
        return value if isinstance(value, bool) else _SKIP
    if isinstance(default, int):
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else _SKIP
    if isinstance(default, float):
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else _SKIP
    if isinstance(default, str):
        return value if isinstance(value, str) else _SKIP
    if isinstance(default, list):
        return [str(v) for v in value] if isinstance(value, list) else _SKIP
    if default is None:  # seed
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None
    return _SKIP


def config_to_json(cfg, include_seed=False):
    """Seed по умолчанию НЕ сохраняется: с seed перестановку можно воспроизвести."""
    d = asdict(cfg)
    if not include_seed:
        d["seed"] = None
    return json.dumps(d, ensure_ascii=False, indent=2)


def config_for_manifest(cfg):
    """Конфигурация для отчётов/манифестов: без seed и локальных путей."""
    d = asdict(cfg)
    d["seed_mode"] = "fixed (debug)" if d.pop("seed") is not None else "random"
    d.pop("arx_jar", None)
    return d


def config_hash(cfg):
    blob = json.dumps(config_for_manifest(cfg), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def config_from_dict(data):
    """Возвращает dict только с известными полями и корректными типами."""
    defaults = Config()
    out = {}
    for f in fields(Config):
        if f.name not in data:
            continue
        val = _coerce(getattr(defaults, f.name), data[f.name])
        if val is not _SKIP:
            out[f.name] = val
    return out


def config_from_json(text):
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Конфигурация должна быть JSON-объектом.")
    return config_from_dict(data)

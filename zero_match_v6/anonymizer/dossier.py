# -*- coding: utf-8 -*-
"""
Карточка обезличивания (досье операции) — то, что Приказ РКН № 140 (Прил. 1, п. 1(2), 1(3), 1(6))
требует определить ДО обезличивания: состав ПДн, субъекты, основание, оценка достаточности метода,
места хранения исходного массива и результата.

Карточка — внутренний документ. Инструмент подсказывает и проверяет полноту, но выводы
(достаточность метода, правовое основание) делает человек.
"""
import hashlib
import json
from copy import deepcopy

LEGAL_BASES = [
    "Согласие субъекта (ст. 6 ч. 1 п. 1)",
    "Договор с субъектом (ст. 6 ч. 1 п. 5)",
    "Статистические или иные цели при условии обезличивания (ст. 6 ч. 1 п. 9)",
    "Законные интересы оператора или третьих лиц",
    "Исполнение требований закона / полномочий",
    "Иное (описать ниже)",
]
SUBJECT_CATEGORIES = ["Клиенты", "Сотрудники", "Соискатели", "Контрагенты (физлица)", "Посетители", "Иные"]
PD_CATEGORIES = ["Общие", "Специальные", "Биометрические", "Иные (платёжные и др.)"]
CONCLUSIONS = ["", "Метод достаточен", "Метод недостаточен", "Требуется доработка"]

# соответствие шагов конвейера методам Приложения № 2 к Приказу № 140 (номера — по тексту Прил. 1, п. 2)
METHODS = {
    "identifiers": "Введение идентификаторов",
    "semantics": "Изменение состава или семантики (п. 4 Прил. 2)",
    "shuffle": "Перемешивание (п. 10 Прил. 2)",
    "decomposition": "Декомпозиция (п. 11 Прил. 2)",
    "generalization": "Обобщение (п. 12 Прил. 2)",
}


def empty_card():
    return {
        "operator_name": "", "responsible_person": "", "dpo_contact": "",
        "purpose": "", "legal_basis": "", "legal_basis_details": "",
        "subject_categories": [], "pd_categories": [], "special_categories_present": False,
        "data_composition": [],             # [{column, role, treatment}]
        "source_location": "", "result_location": "", "additional_info_location": "не сохраняется",
        "recipients": "", "retention_period": "",
        "sufficiency": {"assessed_by": "", "assessed_on": "", "conclusion": "", "rationale": ""},
        "notes": "",
    }


def normalize_card(card):
    """Дополняет карточку недостающими полями (старые версии/неполный JSON)."""
    base = empty_card()
    base.update({k: v for k, v in (card or {}).items() if k != "sufficiency"})
    base["sufficiency"].update((card or {}).get("sufficiency", {}))
    return base


def treatments(col, cfg):
    t = []
    if col in cfg.id_cols:
        t.append("введение идентификаторов")
    if col in cfg.cat_cols or col in cfg.kanon_cols:
        t.append("обобщение")
    if col in cfg.derange_cols:
        t.append("перемешивание" + (" + шум" if (cfg.noise_pct > 0 or cfg.noise_mode == "ldp") else ""))
    if col in cfg.text_cols:
        t.append("изменение состава (удаление идентифицирующих фрагментов)")
    if col == cfg.markov_col:
        t.append("изменение семантики (синтетическая замена)")
    if col in cfg.quasi_cols and cfg.supp_mode != "none":
        t.append(f"подавление малых групп ({cfg.supp_mode})")
    return t or ["не обрабатывается"]


def autofill_composition(inventory, cfg):
    """Состав массива по инвентаризации (роли — эвристика, подтвердите вручную)."""
    return [{"column": r["column"], "role": r["role"], "treatment": "; ".join(treatments(r["column"], cfg))}
            for _, r in inventory.iterrows()]


def sufficiency_draft(summary, attack):
    """Черновик обоснования из метрик прогона. Вывод о достаточности остаётся за человеком."""
    lines = [f"По результатам прогона: минимальный размер группы по QI — {summary['min_k']} "
             f"(цель {summary['k_target']}); остаточных ПДн высокой уверенности — {summary['residual_high']}; "
             f"подавлено записей — {summary['suppression_pct']}%."]
    if summary.get("sensitive_evaluated"):
        lines.append(f"l-разнообразие: min l={summary['l_min']} (цель {summary['l_target']}); "
                     f"t-близость: max t={summary['t_max']} (цель {summary['t_target']}).")
    if attack:
        lines.append(f"Моделирование атаки связывания (знание исходных QI): однозначно найдено "
                     f"{attack['unique_share']}% записей, ожидаемая доля верных сопоставлений — "
                     f"{attack['expected_reid']}%.")
    lines.append("Оценка не учитывает внешние источники данных, которыми может обладать получатель; "
                 "учтите это при выводе о достаточности метода.")
    return "\n".join(lines)


def validate_card(card):
    """Список замечаний; пустой список = карточка заполнена."""
    c = normalize_card(card)
    s = c["sufficiency"]
    issues = []
    required = [("operator_name", "наименование оператора"), ("responsible_person", "ответственный за обезличивание"),
                ("purpose", "цель обработки обезличенных данных"), ("legal_basis", "правовое основание"),
                ("source_location", "место хранения исходного массива"),
                ("result_location", "место хранения обезличенных данных")]
    issues += [f"Не заполнено: {label}" for key, label in required if not str(c[key]).strip()]
    if not c["subject_categories"]:
        issues.append("Не указаны категории субъектов")
    if not c["pd_categories"]:
        issues.append("Не указаны категории ПДн")
    if not c["data_composition"]:
        issues.append("Не определён состав обезличиваемых данных")
    src, res = c["source_location"].strip().lower(), c["result_location"].strip().lower()
    if src and res and src == res:
        issues.append("Исходный массив и результат хранятся в одном месте — совместное хранение недопустимо")
    for key, label in (("assessed_by", "кто провёл оценку достаточности"), ("assessed_on", "дата оценки"),
                       ("rationale", "обоснование достаточности")):
        if not str(s[key]).strip():
            issues.append(f"Оценка достаточности: не заполнено — {label}")
    if s["conclusion"] == "Метод недостаточен":
        issues.append("Оценка достаточности: метод признан недостаточным")
    elif s["conclusion"] == "Требуется доработка":
        issues.append("Оценка достаточности: требуется доработка")
    elif s["conclusion"] != "Метод достаточен":
        issues.append("Оценка достаточности: вывод не выбран")
    return issues


def card_warnings(card):
    c = normalize_card(card)
    w = []
    if "Специальные" in c["pd_categories"] or "Биометрические" in c["pd_categories"] \
            or c["special_categories_present"]:
        w.append("Есть специальные/биометрические категории — у них отдельные условия обработки; "
                 "проверьте основание с юристом.")
    if c["additional_info_location"].strip().lower() not in ("не сохраняется", "нет", "-", ""):
        if c["additional_info_location"].strip().lower() in (c["source_location"].strip().lower(),
                                                              c["result_location"].strip().lower()):
            w.append("Дополнительная информация (соответствия) лежит рядом с данными — это снижает защиту.")
    return w


def card_to_json(card):
    return json.dumps(normalize_card(card), ensure_ascii=False, indent=2, sort_keys=True)


def card_hash(card):
    blob = json.dumps(normalize_card(card), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def copy_card(card):
    return deepcopy(normalize_card(card))

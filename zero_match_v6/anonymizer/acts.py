# -*- coding: utf-8 -*-
"""
Черновики локальных актов по Приказу РКН № 140 (Прил. 1, п. 2) и сопроводительная записка получателю.

Акты — ВНУТРЕННИЕ документы: хранятся отдельно от обезличенных данных, получателю не передаются.
Это черновики для юриста/ответственного за ПДн; инструмент не подменяет их утверждение.
"""
import io
import json
import zipfile
from datetime import datetime

from . import __version__
from .config import NONE_OPTION
from .dossier import METHODS, card_warnings, normalize_card, treatments, validate_card
from .report import INTERNAL_BANNER

DRAFT = "> ЧЕРНОВИК, сформирован автоматически. Требует проверки и утверждения оператором."


def _j(items):
    return ", ".join(items) if items else "—"


def _run_id(res):
    try:
        return json.loads(res["audit_json"])["run_id"]
    except Exception:
        return "—"


def _table(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def act_procedure(card, cfg, res, operator_user):
    c = normalize_card(card)
    comp = _table([(r["column"], r["role"], r["treatment"]) for r in c["data_composition"]],
                  ["Атрибут", "Роль (эвристика)", "Обработка"])
    iso = res["storage_mode"] == "isolated"
    return f"""{INTERNAL_BANNER}

# Порядок обезличивания персональных данных

{DRAFT}

## 1. Общие положения
- Оператор: {c['operator_name'] or '—'}
- Ответственный за обезличивание: {c['responsible_person'] or '—'}; контакт ответственного за ПДн: {c['dpo_contact'] or '—'}
- Цель обработки обезличенных данных: {c['purpose'] or '—'}
- Правовое основание: {c['legal_basis'] or '—'}. {c['legal_basis_details']}
- Получатели обезличенных данных: {c['recipients'] or '—'}; срок хранения: {c['retention_period'] or '—'}

## 2. Массив данных, подлежащих обезличиванию
- Категории субъектов: {_j(c['subject_categories'])}
- Категории ПДн: {_j(c['pd_categories'])}
- Записей в массиве: {res['rows_in']}

{comp}

## 3. Применяемые методы
{_j([METHODS[k] for k in _methods_used(cfg)])}
Правила по каждому методу — в акте «Правила по методам».

## 4. Информационная система и безопасность
- Программа: Zero-Match Enterprise Anonymizer v{__version__} (запуск пользователем {operator_user or '—'}).
- Режим хранения: **{'isolated (изолированный)' if iso else res['storage_mode'] + ' — НЕ изолирован'}**.
{'- Исходный массив и результат не хранятся совместно: исходные данные удалены из сессии приложения после запуска.' if iso else '- ВНИМАНИЕ: режим не изолирован — исходные и обезличенные данные находились в одной сессии. Для реальных ПДн это не соответствует требованию об исключении совместного хранения.'}
- Место хранения исходного массива: {c['source_location'] or '—'}
- Место хранения обезличенных данных: {c['result_location'] or '—'}
- Дополнительная информация (соответствия): {c['additional_info_location'] or '—'}

## 5. Доступ
Доступ третьих лиц к настоящему акту, к сведениям о методах, используемой программе и процедуре обезличивания исключён.
Акты хранятся отдельно от обезличенных данных.

## 6. Учёт действий
Каждая операция фиксируется в журнале рабочего пространства и в протоколе запуска (см. «Карточку учёта операции»).
"""


def _methods_used(cfg):
    used = []
    if cfg.id_cols:
        used.append("identifiers")
    if cfg.text_cols or cfg.markov_col != NONE_OPTION or cfg.supp_mode != "none" or cfg.noise_pct > 0 \
            or cfg.noise_mode == "ldp":
        used.append("semantics")
    if cfg.derange_cols:
        used.append("shuffle")
    if cfg.kanon_cols or cfg.cat_cols:
        used.append("generalization")
    return used


def act_sufficiency(card, cfg, res):
    c = normalize_card(card)
    s, att = res["summary"], res["attack"]
    rows = [("Минимальный размер группы по QI", f"{s['min_k']} (цель {s['k_target']})"),
            ("l-разнообразие, min l", f"{s['l_min']} (цель {s['l_target']})" if s["sensitive_evaluated"] else "не оценивалось"),
            ("t-близость, max t", f"{s['t_max']} (цель {s['t_target']})" if s["sensitive_evaluated"] else "не оценивалось"),
            ("Остаточные ПДн (высокая уверенность)", s["residual_high"]),
            ("Подавлено записей", f"{s['suppression_pct']}%"),
            ("Атака связывания: однозначно найдено", f"{att['unique_share']}%" if att else "не оценивалось"),
            ("Атака связывания: ожидаемый успех", f"{att['expected_reid']}%" if att else "не оценивалось")]
    sf = c["sufficiency"]
    return f"""{INTERNAL_BANNER}

# Оценка достаточности метода обезличивания

{DRAFT}

Оценка проводится на основании категорий субъектов ({_j(c['subject_categories'])}), категорий ПДн
({_j(c['pd_categories'])}) и правового основания ({c['legal_basis'] or '—'}).

## Метрики прогона
{_table(rows, ['Показатель', 'Значение'])}

## Невозможность определения принадлежности без дополнительной информации
Токены генерируются случайно, соответствие «значение → токен» {c['additional_info_location'] or '—'}. Данные выгружаются
в случайном порядке без исходных индексов. Моделирование атаки по квазиидентификаторам даёт показатели выше.
Метрики не учитывают внешние источники, которыми может обладать получатель.

## Вывод (заполняется человеком)
- Оценку провёл: {sf['assessed_by'] or '—'}; дата: {sf['assessed_on'] or '—'}
- Вывод: **{sf['conclusion'] or 'не выбран'}**
- Обоснование: {sf['rationale'] or '—'}

Подпись: ______________________
"""


def act_methods(card, cfg, res):
    bs = res["bin_steps"]
    parts = [f"""{INTERNAL_BANNER}

# Правила по применяемым методам обезличивания

{DRAFT}
"""]
    if cfg.id_cols:
        parts.append(f"""## Введение идентификаторов
Атрибуты: {_j(cfg.id_cols)}. Каждому уникальному значению присваивается случайный токен вида `Client_N`; одинаковые
значения получают одинаковый токен. Таблица соответствий {'НЕ сохраняется' if (card.get('additional_info_location') or 'не сохраняется').strip().lower() == 'не сохраняется' else 'хранится: ' + card['additional_info_location']}.
""")
    sem = []
    if cfg.derange_cols and (cfg.noise_pct > 0 or cfg.noise_mode == "ldp"):
        mode = (f"шум Лапласа с локальной дифференциальной приватностью, ε={cfg.epsilon} на атрибут" if cfg.noise_mode == "ldp"
                else f"шум Лапласа {cfg.noise_pct}% от значения (эвристика)")
        sem.append(f"- Искажение значений ({mode}): {_j(cfg.derange_cols)}")
    if cfg.text_cols:
        sem.append(f"- Удаление идентифицирующих фрагментов свободного текста (e-mail, телефоны, ИНН, СНИЛС, карты, IBAN, "
                   f"паспорта, адреса, даты рождения, имена): {_j(cfg.text_cols)}")
    if cfg.markov_col != NONE_OPTION:
        sem.append(f"- Замена значений синтетическими цепочками (цепь Маркова): {cfg.markov_col}")
    if cfg.supp_mode != "none":
        sem.append(f"- Подавление записей из групп меньше k={cfg.k_target}: режим «{cfg.supp_mode}» "
                   f"(замаскировано {res['suppression']['masked']}, удалено {res['suppression']['dropped']})")
    if sem:
        parts.append("## Изменение состава или семантики (п. 4 Прил. 2)\n" + "\n".join(sem) + "\n")
    if cfg.derange_cols:
        parts.append(f"""## Перемешивание (п. 10 Прил. 2)
Атрибуты: {_j(cfg.derange_cols)}. Алгоритм: значения атрибута случайно переставляются между записями так, что ни одно
значение не остаётся в своей записи (алгоритм Саттоло; единый цикл по всем записям). Источник случайности:
{'фиксированный seed (отладка)' if cfg.seed is not None else 'системный криптостойкий генератор'}.

> **ТРЕБУЕТ РЕШЕНИЯ ОПЕРАТОРА.** Прил. 1, п. 2(4) к Приказу № 140 описывает правила перемешивания как алгоритм
> перестановки значения на заданное оператором количество позиций. Реализованный алгоритм — случайная циклическая
> перестановка, а не сдвиг на фиксированное число позиций. Согласуйте с юристом/ответственным за ПДн, что такое описание
> удовлетворяет требованию, либо измените метод.
""")
    if cfg.kanon_cols or cfg.cat_cols:
        steps = "\n".join(f"- {c}: интервалы шириной {s} (значение заменяется интервалом вида «N-M»)" for c, s in bs.items())
        cats = "\n".join(f"- {c}: значения реже {cfg.rare_min_count} записей объединяются в «Другое»"
                         f"{'; иерархия обобщения задана оператором' if cfg.hierarchy_json.strip() else ''}" for c in cfg.cat_cols)
        parts.append(f"""## Обобщение (п. 12 Прил. 2)
Субъекты, атрибуты которых обобщаются: все записи массива (обобщение применяется ко всем значениям атрибута).
Правила вычисления заменяющих значений и способы группировки:
{steps or '—'}
{cats or ''}
Шаги интервалов {'подобраны автоматически' if cfg.auto_step else 'заданы пресетом'}{' (ручные значения приоритетнее)' if cfg.manual_steps_json.strip() else ''}.
""")
    parts.append("## Декомпозиция (п. 11 Прил. 2)\nНе применяется: массив не разбивается на части.\n")
    return "\n".join(parts)


def act_register(card, cfg, res, operator_user, source_file_sha256=""):
    s = res["summary"]
    rows = [("Идентификатор запуска", _run_id(res)), ("Дата и время", datetime.now().isoformat(timespec="seconds")),
            ("Исполнитель", operator_user or "—"), ("Операция", "Обезличивание массива ПДн"),
            ("Отпечаток исходной таблицы", s["source_fingerprint"]),
            ("SHA-256 исходного файла", source_file_sha256 or "—"),
            ("Записей: на входе → на выходе", f"{s['rows_in']} → {s['rows_out']}"),
            ("SHA-256 выгрузки (CSV)", s["release_sha256"]), ("Хеш конфигурации", s["config_hash"]),
            ("Статус проверок", res["status_code"]), ("Режим хранения", res["storage_mode"]),
            ("Экспорт заблокирован", "да" if s["export_blocked"] else "нет")]
    checks = _table([("✔" if c["passed"] else "✖", c["name"], c["detail"]) for c in res["checks"]],
                    ["", "Проверка", "Результат"])
    return f"""{INTERNAL_BANNER}

# Карточка учёта операции обезличивания

{DRAFT}

{_table(rows, ['Поле', 'Значение'])}

## Результаты проверок
{checks}

Действия с обезличенными данными (передача, удаление, изменение) фиксируются дополнительно в журнале рабочего пространства.
"""


def build_acts(card, cfg, res, operator_user="", source_file_sha256=""):
    """Словарь имя файла → текст (Markdown). Плюс замечания по карточке."""
    card = normalize_card(card)
    issues, warns = validate_card(card), card_warnings(card)
    readme = (INTERNAL_BANNER + "\n\nСодержимое: черновики локальных актов и учёта операции.\n"
              "Хранить отдельно от обезличенных данных; получателю не передавать.\n")
    if res["storage_mode"] != "isolated":
        readme += ("\nВНИМАНИЕ: режим хранения НЕ изолирован — исходные и обезличенные данные находились в одной "
                   "сессии (Приказ № 140, Прил. 1, п. 1(6)).\n")
    if issues:
        readme += "\nНезаполненные пункты карточки:\n" + "\n".join(f"- {i}" for i in issues) + "\n"
    if warns:
        readme += "\nЗамечания:\n" + "\n".join(f"- {w}" for w in warns) + "\n"
    return {
        "00_README_INTERNAL.txt": readme,
        "01_procedure.md": act_procedure(card, cfg, res, operator_user),
        "02_sufficiency_assessment.md": act_sufficiency(card, cfg, res),
        "03_method_rules.md": act_methods(card, cfg, res),
        "04_operation_register.md": act_register(card, cfg, res, operator_user, source_file_sha256),
    }


def build_acts_bundle(card, cfg, res, operator_user="", source_file_sha256=""):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in build_acts(card, cfg, res, operator_user, source_file_sha256).items():
            z.writestr(name, text)
    return buf.getvalue()


def recipient_note(res, file_name="enterprise_safe_data.csv"):
    """Записка получателю: только факты о файле. Методы, параметры и проверки НЕ раскрываются."""
    df = res["df_safe"]
    return (f"СОПРОВОДИТЕЛЬНАЯ ЗАПИСКА\n\nФайл: {file_name}\nДата формирования: {datetime.now():%Y-%m-%d}\n"
            f"Записей: {len(df)}\nКолонок: {len(df.columns)}\nSHA-256 файла: {res['release_sha256']}\n\n"
            "Данные обезличены. Попытки установить личность субъектов, сопоставить записи с другими источниками "
            "и передать данные третьим лицам без согласования с оператором запрещены.\n")

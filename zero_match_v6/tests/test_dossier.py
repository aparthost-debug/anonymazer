# -*- coding: utf-8 -*-
"""Карточка обезличивания, локальные акты, изоляция исходника."""
import io
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anonymizer.acts import build_acts, build_acts_bundle, recipient_note  # noqa: E402
from anonymizer.config import NONE_OPTION, Config  # noqa: E402
from anonymizer.demo import generate_demo_data  # noqa: E402
from anonymizer.dossier import (autofill_composition, card_hash, card_warnings, empty_card,  # noqa: E402
                                normalize_card, sufficiency_draft, validate_card)
from anonymizer.inventory import build_inventory  # noqa: E402
from anonymizer.isolation import ISOLATED, SHARED, purge_source, result_holds_source, storage_mode  # noqa: E402
from anonymizer.pii_detect import suggest_config  # noqa: E402
from anonymizer.pipeline import run_pipeline  # noqa: E402


def _run(n=400, isolated=True, **over):
    df = generate_demo_data(n)
    s = suggest_config(df)
    s["markov_col"] = s["markov_col"] or NONE_OPTION
    s.update(over)
    cfg = Config(**s)
    return df, cfg, run_pipeline(df, cfg, "demo", "f" * 64, storage_mode=storage_mode(isolated))


def _filled(df, cfg):
    c = empty_card()
    c.update(operator_name="ООО Пример", responsible_person="Иванов И.И.", purpose="Аналитика",
             legal_basis="Договор с субъектом (ст. 6 ч. 1 п. 5)", subject_categories=["Клиенты"],
             pd_categories=["Общие"], source_location="Контур 1", result_location="Контур 2",
             data_composition=autofill_composition(build_inventory(df), cfg))
    c["sufficiency"].update(assessed_by="DPO", assessed_on="2026-11-23", conclusion="Метод достаточен",
                            rationale="см. метрики")
    return c


def test_card_validation_rules():
    assert len(validate_card(empty_card())) >= 10
    df, cfg, res = _run()
    c = _filled(df, cfg)
    assert validate_card(c) == []
    same = {**c, "result_location": " контур 1 "}
    assert any("совместное хранение" in i for i in validate_card(same))
    bad = json.loads(json.dumps(c))
    bad["sufficiency"]["conclusion"] = "Метод недостаточен"
    assert any("недостаточным" in i for i in validate_card(bad))
    bad["sufficiency"]["conclusion"] = ""
    assert any("вывод не выбран" in i for i in validate_card(bad))
    assert validate_card({"purpose": "x"})                          # неполный/старый JSON не падает
    assert normalize_card(None)["sufficiency"]["assessed_by"] == ""


def test_card_hash_is_stable_and_sensitive():
    df, cfg, res = _run()
    c = _filled(df, cfg)
    assert card_hash(c) == card_hash(json.loads(json.dumps(c)))
    c2 = json.loads(json.dumps(c))
    c2["purpose"] = "Другое"
    assert card_hash(c) != card_hash(c2)


def test_autofill_composition_maps_treatments_and_never_sets_conclusion():
    df, cfg, res = _run()
    comp = {r["column"]: r for r in autofill_composition(build_inventory(df), cfg)}
    assert "введение идентификаторов" in comp["ФИО"]["treatment"]
    assert "перемешивание" in comp["Сумма_покупок"]["treatment"]
    assert comp["Лояльный"]["treatment"] == "не обрабатывается"
    draft = sufficiency_draft(res["summary"], res["attack"])
    assert "минимальный размер группы" in draft and "внешние источники" in draft
    assert empty_card()["sufficiency"]["conclusion"] == ""                    # вывод — только человеком


def test_card_warnings_for_special_categories():
    c = empty_card()
    c["pd_categories"] = ["Специальные"]
    assert card_warnings(c)


def test_acts_contain_required_sections_and_no_source_values():
    df, cfg, res = _run(isolated=True)
    card = _filled(df, cfg)
    acts = build_acts(card, cfg, res, operator_user="analyst_a", source_file_sha256="a" * 64)
    assert set(acts) == {"00_README_INTERNAL.txt", "01_procedure.md", "02_sufficiency_assessment.md",
                         "03_method_rules.md", "04_operation_register.md"}
    for name, text in acts.items():
        assert text.startswith("ВНУТРЕННИЙ ДОКУМЕНТ"), name
    rules = acts["03_method_rules.md"]
    assert "п. 4 Прил. 2" in rules and "п. 10 Прил. 2" in rules and "п. 12 Прил. 2" in rules and "п. 11 Прил. 2" in rules
    assert "ТРЕБУЕТ РЕШЕНИЯ ОПЕРАТОРА" in rules and "Саттоло" in rules          # честная пометка про п. 2(4)
    assert "Возраст" in rules and "интервалы шириной" in rules
    assert "изолирован" in acts["01_procedure.md"] and "ООО Пример" in acts["01_procedure.md"]
    assert res["release_sha256"] in acts["04_operation_register.md"] and "analyst_a" in acts["04_operation_register.md"]
    assert "Метод достаточен" in acts["02_sufficiency_assessment.md"]
    blob = "\n".join(acts.values())
    for leaked in ("Иванов Иван", "89165554433", "7707083893"):
        assert leaked not in blob
    assert "seed" not in blob.lower().replace("seed (отладка)", "")


def test_acts_flag_shared_storage_and_open_card_items():
    df, cfg, res = _run(isolated=False)
    acts = build_acts(empty_card(), cfg, res)
    assert "НЕ изолирован" in acts["00_README_INTERNAL.txt"] or "Не заполнено" in acts["00_README_INTERNAL.txt"]
    assert "не соответствует требованию" in acts["01_procedure.md"]


def test_acts_bundle_is_zip_and_recipient_note_hides_methods():
    df, cfg, res = _run()
    z = zipfile.ZipFile(io.BytesIO(build_acts_bundle(_filled(df, cfg), cfg, res)))
    assert len(z.namelist()) == 5
    note = recipient_note(res)
    assert res["release_sha256"] in note and "запрещены" in note
    for secret in ("Саттоло", "k-анонимность", "шум", "токен", "ε", "метод"):
        assert secret.lower() not in note.lower().replace("методам", "")


def test_purge_removes_source_and_pairing_but_keeps_aggregates():
    df, cfg, res = _run(isolated=True)
    assert result_holds_source(res, df)
    purge_source(res)
    assert not result_holds_source(res, df) and res["purged"] and res["pair_pos"] is None and res["df_raw"] is None
    assert res["dist"] and res["utility"] is not None and res["attack"] and len(res["csv_bytes"]) > 0
    assert ISOLATED != SHARED

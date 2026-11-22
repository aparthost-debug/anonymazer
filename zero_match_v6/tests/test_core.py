# -*- coding: utf-8 -*-
"""Тесты ядра. Запуск: pytest -q   (или python tests/run_tests.py без pytest)."""
import json
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anonymizer.config import (NONE_OPTION, Config, config_for_manifest, config_from_json,  # noqa: E402
                               config_to_json)
from anonymizer.demo import generate_demo_data  # noqa: E402
from anonymizer.metrics import (compute_risk_metrics, ks_statistic, linkage_attack,  # noqa: E402
                                sensitive_diversity, suppression_bias, utility_report)
from anonymizer.optimize import optimize_steps, pareto_steps, suppress_small_groups  # noqa: E402
from anonymizer.pii_detect import (classify_columns, count_pii_cells, detect_pii,  # noqa: E402
                                   suggest_config)
from anonymizer.pipeline import POS_COL, run_pipeline  # noqa: E402
from anonymizer.report import (build_pdf_bytes, bytes_hash, canonical_fingerprint,  # noqa: E402
                               to_excel_bytes)
from anonymizer.text_clean import (card_ok, clean_free_text, clean_series, iban_ok,  # noqa: E402
                                   inn_ok, snils_ok)
from anonymizer.transforms import (add_noise_ldp, add_noise_pct, generalize_categories,  # noqa: E402
                                   generalize_numeric_series, interval_midpoints,
                                   parse_hierarchy, sattolo_derangement, to_numeric_view)


# ---------- очистка текста ----------
def test_checksums():
    assert inn_ok("7707083893") and not inn_ok("7707083894")
    assert inn_ok("500100732259")
    assert snils_ok("11223344595") and not snils_ok("11223344596")
    assert card_ok("4111111111111111") and not card_ok("4111111111111112")


def test_iban_and_inn_context_and_no_word_false_positives():
    assert iban_ok("DE89 3704 0044 0532 0130 00") and not iban_ok("DE89370400440532013001")
    assert "[IBAN УДАЛЕН]" in clean_free_text("счёт DE89 3704 0044 0532 0130 00")
    assert clean_free_text("sku ab12abcdefghijk") == "sku ab12abcdefghijk"
    assert "[ИНН УДАЛЕН]" in clean_free_text("ИНН 1234567890")            # контекст без чексуммы
    assert clean_free_text("заказ 1234567890") == "заказ 1234567890"      # просто номер
    assert "[ТЕЛЕФОН УДАЛЕН]" in clean_free_text("тел +44 20 7946 0958")


def test_classify_columns_uses_tokens_not_substrings():
    d = pd.DataFrame({"Домен_сайта": ["a.ru"] * 30, "Финн": list("ab") * 15,
                      "Телефон": ["89165554433"] * 30, "ФИО": ["Иванов Иван"] * 30,
                      "Сумма": np.linspace(1, 100, 30), "Город": ["Москва"] * 30})
    r = classify_columns(d).set_index("column")["suggested_role"].to_dict()
    assert r["Домен_сайта"] == "ordinary" and r["Финн"] == "ordinary"
    assert r["Телефон"] == "direct_identifier" and r["ФИО"] == "direct_identifier"
    assert r["Сумма"] == "sensitive" and r["Город"] == "quasi_identifier"
    float_ids = pd.DataFrame({"Суммарный_балл": np.arange(30) * 1.5})
    assert classify_columns(float_ids).iloc[0]["suggested_role"] != "direct_identifier"


def test_clean_contacts_and_documents():
    assert clean_free_text("Тел +7 (999) 111-22-33") == "Тел [ТЕЛЕФОН УДАЛЕН]"
    assert "[ТЕЛЕФОН УДАЛЕН]" in clean_free_text("звонить 89165554433")
    assert clean_free_text("a.b@mail.ru") == "[EMAIL УДАЛЕН]"
    assert "[ИНН УДАЛЕН]" in clean_free_text("ИНН 7707083893")
    assert "[СНИЛС УДАЛЕН]" in clean_free_text("СНИЛС 112-233-445 95")
    assert "[КАРТА УДАЛЕНА]" in clean_free_text("карта 4111 1111 1111 1111")
    assert "[ПАСПОРТ УДАЛЕН]" in clean_free_text("паспорт 45 06 123456")
    assert "[АДРЕС УДАЛЕН]" in clean_free_text("живёт: ул. Ленина, д. 5, кв. 12")
    assert "[ДАТА УДАЛЕНА]" in clean_free_text("родился 01.02.1990")


def test_no_false_positive_on_long_numbers():
    assert clean_free_text("Заказ 12345678901234567890") == "Заказ 12345678901234567890"


def test_names_with_declension():
    out = clean_free_text("Подарок Анне от Иванова", names=("Иванов Иван", "Иванов", "Иван", "Смирнова Анна", "Смирнова", "Анна"))
    assert "Анне" not in out and "Иванова" not in out


def test_clean_series_uses_names_per_row():
    s = pd.Series(["Анне скидка", "Анне скидка", None, 5])
    out = clean_series(s, [("Анна",), (), (), ()])
    assert "[ИМЯ УДАЛЕНО]" in out.iloc[0] and out.iloc[1] == "Анне скидка"
    assert out.iloc[2] is None or pd.isna(out.iloc[2])
    assert out.iloc[3] == 5


# ---------- перестановка и шум ----------
def test_sattolo_no_fixed_points():
    x = list(range(100))
    y = sattolo_derangement(x, random.Random(1))
    assert sorted(y) == x and all(a != b for a, b in zip(x, y))


def test_noise_keeps_dtype_and_sign():
    rng = np.random.default_rng(0)
    s = pd.Series([1, 2, 3, 1000, 5000])
    out = add_noise_pct(s, 30, rng)
    assert out.dtype == s.dtype and (out >= 0).all()
    out2, info = add_noise_ldp(s, 1.0, rng, 1.0)
    assert info["scale"] > 0 and (out2 >= 0).all() and out2.dtype == s.dtype
    txt = pd.Series(["a", "b"])
    assert add_noise_pct(txt, 5, rng) is txt


def test_ldp_noise_scales_with_epsilon():
    s = pd.Series(np.linspace(0, 100, 2000))
    d = {}
    for eps in (0.5, 50.0):
        out, _ = add_noise_ldp(s, eps, np.random.default_rng(1), 0.0)
        d[eps] = float(np.abs(out - s).mean())
    assert d[0.5] > 10 * d[50.0]


# ---------- обобщение ----------
def test_generalize_and_midpoints():
    g = generalize_numeric_series(pd.Series([25, 34, None, -5]), 10)
    assert g.iloc[0] == "20-29" and g.iloc[1] == "30-39" and pd.isna(g.iloc[2]) and g.iloc[3] == "-10--1"
    m = interval_midpoints(pd.Series(["20-29", "-10--1", "[20, 30[", "(20; 30)", "abc", 5]))
    assert m.iloc[0] == 24.5 and m.iloc[1] == -5.5 and m.iloc[2] == 25 and m.iloc[3] == 25
    assert m.iloc[4:].isna().all()
    assert to_numeric_view(pd.Series(["20-29", "7"])).tolist() == [24.5, 7.0]


def test_categories_hierarchy_and_rare():
    s = pd.Series(["Москва"] * 6 + ["Тверь", "Казань", None])
    out, fmap = generalize_categories(s, {"Тверь": "Центр"}, min_count=3)
    assert out.iloc[0] == "Москва" and out.iloc[6] == "Другое" and out.iloc[7] == "Другое"
    assert pd.isna(out.iloc[8]) and fmap["Тверь"] == "Другое"
    try:
        parse_hierarchy('["x"]')
        assert False
    except ValueError:
        pass


# ---------- метрики ----------
def test_risk_keeps_nan_rows():
    d = pd.DataFrame({"a": [1, 1, None, None, 2], "b": ["x", "x", "y", "y", "z"]})
    r = compute_risk_metrics(d, ["a", "b"], k=2)
    assert r["unique_share"] == 20.0 and r["below_k_share"] == 20.0 and r["min_k"] == 1


def test_l_diversity_and_t_closeness():
    df = pd.DataFrame({"qi": ["a"] * 4 + ["b"] * 4, "s": ["x", "x", "x", "x", "x", "y", "z", "w"]})
    r = sensitive_diversity(df, ["qi"], ["s"], l_target=2, t_target=0.3).iloc[0]
    assert r["Мин. l"] == 1
    assert r["Записей в классах с l < L (%)"] == 50.0
    assert r["Макс. t"] > 0.3


def test_ks():
    a = np.arange(100)
    assert ks_statistic(a, a) == 0.0
    assert ks_statistic(a, a + 1000) == 1.0


def test_utility_report():
    raw = pd.DataFrame({"x": np.arange(100.0), "y": np.arange(100.0) * 2, "c": ["a", "b"] * 50})
    safe = raw.copy()
    rep, corr = utility_report(raw, safe, ["x", "y", "c"])
    assert (rep["Значение"] == 0).all() and corr == 0.0


# ---------- подавление и подбор шага ----------
def test_suppression_modes():
    df = pd.DataFrame({"qi": ["a"] * 5 + ["b"] * 2 + ["c"] * 3, "v": range(10)})
    d, info = suppress_small_groups(df, ["qi"], 3, "drop")
    assert len(d) == 8 and info["dropped"] == 2
    d, info = suppress_small_groups(df, ["qi"], 3, "mask")
    assert info["masked"] == 0 and info["dropped"] == 2          # «*»-группа (2) < k -> удалена
    df2 = pd.DataFrame({"qi": ["a"] * 5 + ["b"] * 2 + ["c"] * 2, "v": range(9)})
    d, info = suppress_small_groups(df2, ["qi"], 3, "mask")
    assert info["masked"] == 4 and (d.loc[d["qi"] == "*"].shape[0] == 4)
    assert compute_risk_metrics(d, ["qi"], 3)["below_k_share"] == 0.0


def test_optimize_steps_picks_feasible_minimum():
    df = generate_demo_data(500)
    steps, rep = optimize_steps(df, ["Возраст"], ["Возраст"], 5, 0.0)
    assert rep[0]["Условие выполнено"] and steps["Возраст"] in pareto_steps(10)
    binned = df.copy()
    binned["Возраст"] = generalize_numeric_series(binned["Возраст"], steps["Возраст"])
    assert compute_risk_metrics(binned, ["Возраст"], 5)["below_k_share"] == 0.0


# ---------- атака связывания ----------
def test_linkage_attack_unique_rows_are_found():
    raw = pd.DataFrame({"a": ["x", "y", "z", "z"]})
    safe = raw.copy()
    safe[POS_COL] = np.arange(4)
    r = linkage_attack(raw, safe, ["a"], {}, {})
    assert r["unique_share"] == 50.0 and r["correct_unique_share"] == 50.0
    masked = pd.DataFrame({"a": ["*"] * 4, POS_COL: np.arange(4)})
    r = linkage_attack(raw, masked, ["a"], {}, {})
    assert r["unique_share"] == 0.0 and r["avg_candidates"] == 4.0 and r["expected_reid"] == 25.0


# ---------- ПДн ----------
def test_pii_detect_and_suggest():
    df = generate_demo_data(500)
    rep = detect_pii(df)
    assert "Комментарий_менеджера" in rep["Колонка"].values
    s = suggest_config(df, rep)
    assert s["id_cols"] == ["ФИО"] and s["text_cols"] == ["Комментарий_менеджера"]
    assert s["cat_cols"] == ["Город"]
    assert s["derange_cols"] == ["Сумма_покупок"] and s["kanon_cols"] == ["Возраст"]
    assert s["markov_col"] == "История_действий" and "Город" in s["quasi_cols"]


def test_residual_pii_counter_ignores_city_names_and_tiers_numbers():
    df = pd.DataFrame({"city": ["Санкт-Петербург", "Нижний Новгород"], "t": ["ok", "a@b.ru"]})
    r = count_pii_cells(df)
    assert r["total"] == 1 and list(r["by_column"]) == ["t"] and r["low_total"] == 0
    d2 = pd.DataFrame({"a": ["1234567890"] * 3, "sku": ["ab12abcdefghijk"] * 3, "b": ["7707083893"] * 2 + ["x"]})
    r2 = count_pii_cells(d2)
    assert r2["total"] == 2 and list(r2["by_column"]) == ["b"] and r2["low_total"] == 3


# ---------- конфигурация ----------
def test_config_roundtrip_and_bad_values():
    cfg = Config(id_cols=["a"], noise_mode="ldp", epsilon=2.0, seed=7)
    back = config_from_json(config_to_json(cfg))
    assert back["id_cols"] == ["a"] and back["epsilon"] == 2.0 and back["seed"] is None
    assert config_from_json(config_to_json(cfg, include_seed=True))["seed"] == 7
    bad = config_from_json(json.dumps({"k_target": "много", "unknown": 1, "epsilon": 3}))
    assert "k_target" not in bad and "unknown" not in bad and bad["epsilon"] == 3.0


# ---------- конвейер ----------
def _cfg_for(df, **over):
    s = suggest_config(df)
    s["markov_col"] = s["markov_col"] or NONE_OPTION
    s.update(over)
    return Config(**s)


def test_pipeline_end_to_end():
    df = generate_demo_data(500)
    r = run_pipeline(df, _cfg_for(df), "demo", "0" * 64)
    out = r["df_safe"]
    assert POS_COL not in out.columns and len(out) == 500 and list(out.index) == list(range(500))
    tokens = out["ФИО"]
    assert tokens.str.startswith("Client_").all() and tokens.nunique() == df["ФИО"].nunique()
    assert r["status_code"] in ("ok", "ok_warn") and not r["export_blocked"]
    assert r["after"]["below_k_share"] == 0.0 and r["residual"]["total"] == 0
    assert out["Сумма_покупок"].min() >= 0
    assert not out["Комментарий_менеджера"].str.contains("@|8916|7707083893|Ленина").any()
    assert r["attack"]["unique_share"] == 0.0
    assert "Другое" in set(out["Город"]) or (out["Город"] == "*").any()
    assert "ОТЧЁТ ОБ ОБЕЗЛИЧИВАНИИ" in r["cert_text"]


def test_release_hash_matches_exported_bytes_and_no_leaks():
    df = generate_demo_data(300)
    cfg = _cfg_for(df, seed=123)
    r = run_pipeline(df, cfg, "d", "")
    assert r["release_sha256"] == bytes_hash(r["csv_bytes"]) == r["summary"]["release_sha256"]
    assert r["csv_bytes"].decode("utf-8-sig") == r["df_safe"].to_csv(index=False)
    audit = json.loads(r["audit_json"])
    assert "123" not in json.dumps(audit["config"]) and audit["config"]["seed_mode"] == "fixed (debug)"
    assert "seed" not in audit["config"]
    assert "Иванов" not in r["audit_json"] and "pair_pos" not in r["audit_json"]
    assert len(r["pair_pos"]) == len(r["df_safe"]) and set(r["pair_pos"]) <= set(range(300))
    assert json.loads(config_to_json(cfg))["seed"] is None            # seed не сохраняется
    assert "arx_jar" not in config_for_manifest(cfg)


def test_direct_identifier_left_untouched_blocks_export():
    df = generate_demo_data(200)
    df["Телефон"] = "89165554433"
    r = run_pipeline(df, _cfg_for(df, id_cols=["ФИО"]), "d", "")
    assert "Телефон" in r["direct_unhandled"] and r["export_blocked"]
    assert r["status_code"] == "needs_work"


def test_low_confidence_numbers_warn_but_do_not_block():
    df = generate_demo_data(200)
    df["Заказ"] = "1234567890"
    r = run_pipeline(df, _cfg_for(df), "d", "")
    assert r["residual"]["total"] == 0 and r["residual"]["low_total"] == len(r["df_safe"])
    assert r["status_code"] in ("ok_warn", "needs_work") and "Заказ" not in r["direct_unhandled"]
    assert not any(c["name"].startswith("Необработанные") and not c["passed"] for c in r["checks"])


def test_manual_steps_override_auto():
    df = generate_demo_data(300)
    r = run_pipeline(df, _cfg_for(df, manual_steps_json='{"Возраст": 20}'), "d", "")
    assert r["bin_steps"]["Возраст"] == 20
    r2 = run_pipeline(df, _cfg_for(df, manual_steps_json="не json"), "d", "")
    assert any("Ручные шаги" in n for n in r2["notes"])


def test_enforce_flags_control_blocking():
    df = generate_demo_data(500)
    soft = run_pipeline(df, _cfg_for(df, enforce_t=False, t_target=0.01), "d", "")
    hard = run_pipeline(df, _cfg_for(df, enforce_t=True, t_target=0.01), "d", "")
    assert soft["status_code"] == "ok_warn" and not soft["export_blocked"]
    assert hard["status_code"] == "needs_work" and hard["export_blocked"]


def test_suppression_bias_reports_dropped_categories():
    raw = pd.DataFrame({"c": ["a"] * 8 + ["b"] * 2})
    b = suppression_bias(raw, ["c"], np.arange(8))
    assert b.iloc[0]["Больше всего удалено из"] == "b" and b.iloc[0]["Удалено из неё (%)"] == 100.0


def test_canonical_fingerprint_ignores_index_and_dtype_internals():
    a = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    b = a.iloc[::-1].reset_index(drop=True).iloc[::-1].reset_index(drop=True)
    b.index = [10, 11]
    assert canonical_fingerprint(a) == canonical_fingerprint(b)


def test_pipeline_row_order_is_shuffled_and_tokens_random():
    df = generate_demo_data(500)
    out = run_pipeline(df, _cfg_for(df), "d", "")["df_safe"]
    tokens = out["ФИО"].str.replace("Client_", "").astype(int)
    assert not tokens.is_monotonic_increasing


def test_pipeline_ldp_reports_epsilon():
    df = generate_demo_data(500)
    r = run_pipeline(df, _cfg_for(df, noise_mode="ldp", epsilon=2.0), "d", "")
    assert "Сумма_покупок" in r["dp_info"] and "суммарно ε=2.0" in r["cert_text"]


def test_pipeline_needs_work_when_k_unreachable_without_suppression():
    df = generate_demo_data(500)
    cfg = _cfg_for(df, k_target=400, supp_mode="none", auto_step=False)
    r = run_pipeline(df, cfg, "d", "")
    assert r["status_code"] == "needs_work"


def test_pipeline_fixed_seed_is_reproducible():
    df = generate_demo_data(100)
    a = run_pipeline(df, _cfg_for(df, seed=5), "d", "")["df_safe"]
    b = run_pipeline(df, _cfg_for(df, seed=5), "d", "")["df_safe"]
    pd.testing.assert_frame_equal(a, b)


def test_pipeline_no_qi_is_not_checked():
    df = generate_demo_data(100)
    r = run_pipeline(df, _cfg_for(df, quasi_cols=[]), "d", "")
    assert r["status_code"] == "not_checked" and r["attack"] is None


def test_pipeline_drop_mode_removes_rows_and_attack_counts_them():
    df = generate_demo_data(500)
    r = run_pipeline(df, _cfg_for(df, k_target=60, supp_mode="drop", auto_step=False), "d", "")
    assert len(r["df_safe"]) < 500 and r["attack"]["dropped_share"] > 0
    assert r["after"]["below_k_share"] == 0.0


# ---------- экспорт ----------
def test_exports():
    df = generate_demo_data(20)
    assert to_excel_bytes(df)[:2] == b"PK"
    try:
        assert build_pdf_bytes("Отчёт\nстрока")[:4] == b"%PDF"
    except RuntimeError:
        pass  # нет reportlab или шрифта — допустимо


def test_storage_mode_banner_and_aggregates():
    df = generate_demo_data(300)
    r = run_pipeline(df, _cfg_for(df), "d", "", storage_mode="isolated")
    assert r["summary"]["storage_mode"] == "isolated" and json.loads(r["audit_json"])["storage_mode"] == "isolated"
    assert r["cert_text"].startswith("ВНУТРЕННИЙ ДОКУМЕНТ") and json.loads(r["audit_json"])["classification"].startswith("ВНУТРЕННИЙ")
    d = r["dist"]["Возраст"]
    assert d["type"] == "numeric" and abs(sum(d["before"]) - 1) < 1e-3
    c = r["dist"]["Лояльный"]
    assert c["type"] == "categorical" and set(c["labels"]) == {"да", "нет"}
    assert run_pipeline(df, _cfg_for(df), "d", "")["storage_mode"] == "unspecified"

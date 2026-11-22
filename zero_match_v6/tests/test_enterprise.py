# -*- coding: utf-8 -*-
"""Политики, доказательства, инвентаризация, бенчмарк."""
import copy
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anonymizer import signing  # noqa: E402
from anonymizer.benchmark import (analytical_utility_benchmark, optimize_generalization)  # noqa: E402
from anonymizer.config import NONE_OPTION, Config, config_for_manifest  # noqa: E402
from anonymizer.demo import generate_demo_data  # noqa: E402
from anonymizer.evidence import (build_manifest, manifest_pdf_bytes, verify_audit_chain,  # noqa: E402
                                 verify_manifest)
from anonymizer.inventory import PROFILES, build_enterprise_report, build_inventory, risk_assessment  # noqa: E402
from anonymizer.pii_detect import suggest_config  # noqa: E402
from anonymizer.pipeline import run_pipeline  # noqa: E402
from anonymizer.dossier import empty_card  # noqa: E402
from anonymizer.policy import SCENARIOS, build_policy_artifact, evaluate_policy  # noqa: E402


def _run(n=500, **over):
    df = generate_demo_data(n)
    s = suggest_config(df)
    s["markov_col"] = s["markov_col"] or NONE_OPTION
    s.update(over)
    cfg = Config(**s)
    return df, cfg, run_pipeline(df, cfg, "demo", "0" * 64, storage_mode="isolated")


# ---------- инвентаризация ----------
def test_inventory_has_no_raw_values_and_is_fast():
    df = generate_demo_data(200)
    inv = build_inventory(df)
    assert "sample_preview" not in inv.columns
    blob = inv.to_json(force_ascii=False) + json.dumps(risk_assessment(df, ["Город"]), ensure_ascii=False, default=str)
    assert "Иванов" not in blob and "89165554433" not in blob
    wide = pd.DataFrame(np.random.default_rng(0).integers(0, 100, (500, 60)),
                        columns=[f"c{i}" for i in range(60)])
    build_inventory(wide)


def test_risk_assessment_levels_and_profiles():
    df = generate_demo_data(300)
    r = risk_assessment(df, ["Город", "Возраст"])
    assert r["level"] in ("low", "medium", "high") and r["pii_high_cells"] > 0
    assert PROFILES["Public Release"]["k_target"] > PROFILES["Internal BI"]["k_target"]
    cfg = Config()
    rep = build_enterprise_report(build_inventory(df), r, "Internal BI", cfg)
    assert "seed" not in rep["configuration"]


# ---------- политики ----------
def test_policy_scenarios_differ_and_use_real_results():
    df, cfg, res = _run()
    s = res["summary"]
    d = {name: evaluate_policy(name, s)["decision"] for name in SCENARIOS}
    assert d["Internal BI"] == "NEEDS_ATTESTATION"
    assert d["Public Release"] == "BLOCKED"            # k=5 < 20 и т.д.
    checks = {n: [c["control"] for c in evaluate_policy(n, s)["checks"]] for n in SCENARIOS}
    assert len({json.dumps(v) for v in checks.values()}) == len(SCENARIOS)


def test_policy_no_run_and_attestations_cannot_override_failures():
    assert evaluate_policy("Internal BI", None)["decision"] == "NO_RUN"
    df, cfg, res = _run()
    all_att = {a: True for sc in SCENARIOS.values() for a in sc["attestations"]}
    assert evaluate_policy("Public Release", res["summary"], approvals=all_att)["decision"] == "BLOCKED"
    ready = evaluate_policy("Internal BI", res["summary"], approvals=all_att)
    assert ready["decision"] == "READY_FOR_REVIEW"


def test_policy_blocks_when_direct_identifier_unhandled():
    df = generate_demo_data(200)
    df["Телефон"] = "89165554433"
    s = suggest_config(df)
    s["id_cols"] = ["ФИО"]
    s["markov_col"] = s["markov_col"] or NONE_OPTION
    res = run_pipeline(df, Config(**s), "d", "", storage_mode="isolated")
    ev = evaluate_policy("Internal BI", res["summary"], approvals={a: True for a in SCENARIOS["Internal BI"]["attestations"]})
    assert ev["decision"] == "BLOCKED"
    assert build_policy_artifact(ev)["artifact_sha256"]


# ---------- доказательства ----------
def test_manifest_hash_chain_and_tampering(tmp_path=None):
    df, cfg, res = _run(300)
    m = build_manifest(res["summary"], config_for_manifest(cfg), "f" * 64)
    v = verify_manifest(m)
    assert v["hash_ok"] and v["chain_ok"] and v["signature"] == "absent"
    stamps = [e["timestamp_utc"] for e in m["audit_chain"]]
    assert len(set(stamps)) == len(stamps)                      # у каждого события своё время
    bad = copy.deepcopy(m)
    bad["pipeline"]["min_k"] = 999
    assert not verify_manifest(bad)["hash_ok"]
    bad2 = copy.deepcopy(m)
    bad2["audit_chain"][1]["payload"]["status"] = "ok"
    assert not verify_audit_chain(bad2["audit_chain"])[0]
    assert "seed" not in m["configuration"] and "Иванов" not in json.dumps(m, ensure_ascii=False)
    assert m["release"]["csv_sha256"] == res["release_sha256"]


def test_manifest_signature_trusted_vs_embedded_key():
    import tempfile
    if not signing.crypto_available():
        return
    d = tempfile.mkdtemp()
    signer = signing.load_signer(os.path.join(d, "k.pem"), create=True)
    other = signing.load_signer(os.path.join(d, "other.pem"), create=True)
    df, cfg, res = _run(200)
    m = build_manifest(res["summary"], config_for_manifest(cfg), "", signer=signer)
    assert verify_manifest(m)["signature"] == "valid_untrusted"
    assert verify_manifest(m, signer.public_pem)["signature"] == "valid_trusted"
    assert verify_manifest(m, other.public_pem)["signature"] == "valid_untrusted"
    forged = copy.deepcopy(m)
    forged["signature"]["signature_b64"] = m["signature"]["signature_b64"][:-4] + "AAAA"
    assert verify_manifest(forged)["signature"] == "invalid"
    assert manifest_pdf_bytes(m)[:4] == b"%PDF"


# ---------- бенчмарк ----------
def _benchmark_df():
    rng = np.random.default_rng(0)
    df = generate_demo_data(600)
    df["Сегмент"] = np.where(df["Возраст"] + rng.normal(0, 4, len(df)) > 42, "A", "B")
    return df


def test_benchmark_is_paired_and_detects_signal():
    df = _benchmark_df()
    s = suggest_config(df)
    s.update(markov_col=NONE_OPTION, id_cols=["ФИО"], sensitive_cols=[])
    res = run_pipeline(df, Config(**s), "d", "")
    b = analytical_utility_benchmark(df, res["df_safe"], res["pair_pos"], "Сегмент",
                                     ["Возраст", "Город"], n_estimators=40)
    assert b["task"] == "classification" and b["informative"] and b["coverage_pct"] == 100.0
    assert b["level"] in ("Excellent", "Good", "Moderate", "Poor")
    # цель изменена -> корректный отказ
    try:
        analytical_utility_benchmark(df, res["df_safe"], res["pair_pos"], "Возраст", ["Город"], n_estimators=10)
        assert False
    except ValueError as e:
        assert "изменена" in str(e)


def test_benchmark_flags_uninformative_target():
    df = generate_demo_data(600)
    df["Шум"] = np.random.default_rng(1).choice(["x", "y"], len(df))
    s = suggest_config(df)
    s.update(markov_col=NONE_OPTION)
    res = run_pipeline(df, Config(**s), "d", "")
    b = analytical_utility_benchmark(df, res["df_safe"], res["pair_pos"], "Шум", ["Город"], n_estimators=20)
    assert not b["informative"] and b["note"]


def test_optimizer_recommends_feasible_candidate():
    df = _benchmark_df()
    out = optimize_generalization(df, ["Город", "Возраст"], "Сегмент", ["Возраст", "Город"],
                                  k=5, max_candidates=6, n_estimators=20)
    t = out["table"]
    assert len(t) <= 6 and t["Парето"].any()
    rec = out["recommended"]
    assert "Возраст" in rec["manual_steps"]
    if rec["feasible"]:
        row = t[t["Записей в группах < k (%)"] == 0]
        assert not row.empty


def _card():
    c = empty_card()
    c.update(operator_name="ООО", responsible_person="И.И.", purpose="BI", legal_basis="Договор с субъектом (ст. 6 ч. 1 п. 5)",
             subject_categories=["Клиенты"], pd_categories=["Общие"], source_location="A", result_location="B",
             data_composition=[{"column": "x", "role": "ordinary", "treatment": "-"}])
    c["sufficiency"].update(assessed_by="DPO", assessed_on="2026-11-23", conclusion="Метод достаточен", rationale="ok")
    return c


def test_policy_requires_isolated_storage_and_card():
    df, cfg, res = _run(500)
    s = res["summary"]
    shared = {**s, "storage_mode": "shared (dev)"}
    att = {a: True for a in SCENARIOS["External Vendor"]["attestations"]}
    ev = evaluate_policy("External Vendor", shared, approvals=att, card=_card())
    assert ev["decision"] == "BLOCKED" and any("isolated" in c["control"] and c["status"] == "fail" for c in ev["checks"])
    ev2 = evaluate_policy("External Vendor", s, approvals=att, card=None)
    assert any("card" in c["control"] and c["status"] == "fail" for c in ev2["checks"])
    # Internal BI: карточка рекомендована (warn), но не блокирует
    ev3 = evaluate_policy("Internal BI", s, approvals={a: True for a in SCENARIOS["Internal BI"]["attestations"]})
    assert ev3["decision"] == "READY_FOR_REVIEW" and any(c["status"] == "warn" for c in ev3["checks"])

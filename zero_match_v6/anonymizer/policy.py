# -*- coding: utf-8 -*-
"""Политики выпуска: автоматические проверки по результату запуска + ручные подтверждения."""
import hashlib
import json

from .dossier import validate_card
from .report import json_default

SCENARIOS = {
    "Internal BI": {
        "description": "Внутренняя аналитика, ограниченный круг.",
        "min_k": 5, "require_l_t": False, "min_l": 2, "max_t": 1.0, "max_suppression_pct": 20,
        "require_benchmark": False, "require_card": False,
        "attestations": ["Чувствительные атрибуты просмотрены", "Получатели определены"],
    },
    "ML / Research": {
        "description": "Обучение моделей и исследования.",
        "min_k": 5, "require_l_t": True, "min_l": 2, "max_t": 0.35, "max_suppression_pct": 10,
        "require_benchmark": True, "require_card": True,
        "attestations": ["Целевая переменная не содержит ПДн", "Чувствительные атрибуты просмотрены"],
    },
    "External Vendor": {
        "description": "Передача внешнему подрядчику.",
        "min_k": 10, "require_l_t": True, "min_l": 3, "max_t": 0.25, "max_suppression_pct": 10,
        "require_benchmark": False, "require_card": True,
        "attestations": ["Договор и цель обработки подтверждены", "Получатель согласован",
                         "Чувствительные атрибуты просмотрены"],
    },
    "Public Release": {
        "description": "Публикация без ограничений получателей.",
        "min_k": 20, "require_l_t": True, "min_l": 3, "max_t": 0.15, "max_suppression_pct": 5,
        "require_benchmark": True, "require_card": True,
        "attestations": ["Экспертная оценка риска проведена", "Юридическое согласование получено",
                         "Чувствительные атрибуты просмотрены"],
    },
}


def evaluate_policy(scenario, summary, benchmark=None, approvals=None, card=None):
    """
    decision: NO_RUN — нет результата запуска; BLOCKED — не пройдена автоматическая проверка;
    NEEDS_ATTESTATION — авто-проверки пройдены, не все ручные подтверждения;
    READY_FOR_REVIEW — всё пройдено (не равно «разрешено к выпуску»).
    """
    sc = SCENARIOS[scenario]
    approvals = approvals or {}
    if not summary:
        return {"scenario": scenario, "decision": "NO_RUN", "checks": [], "missing_attestations": [],
                "description": sc["description"]}
    checks = []

    def add(control, passed, detail):
        checks.append({"control": control, "status": "pass" if passed else "fail",
                       "auto": True, "detail": detail})

    add("Pipeline status", summary["status_code"] in ("ok", "ok_warn"), summary["status_code"])
    add("Source isolated from result (Order 140, Annex 1, p.1(6))",
        summary.get("storage_mode") == "isolated", str(summary.get("storage_mode", "unspecified")))
    issues = validate_card(card) if card else ["карточка обезличивания не заполнена"]
    if sc["require_card"]:
        add("Depersonalization card complete (Order 140, Annex 1, p.1(2),(3))", not issues,
            "ok" if not issues else "; ".join(issues[:3]) + (" …" if len(issues) > 3 else ""))
    else:
        checks.append({"control": "Depersonalization card (recommended)", "status": "pass" if not issues else "warn",
                       "auto": True, "detail": "ok" if not issues else f"{len(issues)} open item(s)"})
    add("Export not blocked", not summary["export_blocked"], "blocked" if summary["export_blocked"] else "ok")
    add(f"Minimum k >= {sc['min_k']}", summary["min_k"] >= sc["min_k"], f"min_k={summary['min_k']}")
    if sc["require_l_t"]:
        add("Sensitive attributes evaluated", summary["sensitive_evaluated"],
            "l/t considered" if summary["sensitive_evaluated"] else "no sensitive columns selected")
        if summary["sensitive_evaluated"]:
            add(f"l-diversity >= {sc['min_l']}", summary["l_min"] >= sc["min_l"], f"l_min={summary['l_min']}")
            add(f"t-closeness <= {sc['max_t']}", summary["t_max"] <= sc["max_t"], f"t_max={summary['t_max']}")
    add("No residual PII (high confidence)", summary["residual_high"] == 0, f"cells={summary['residual_high']}")
    add("Direct identifiers handled", not summary["direct_unhandled"],
        ", ".join(summary["direct_unhandled"]) or "none")
    add(f"Suppression <= {sc['max_suppression_pct']}%",
        summary["suppression_pct"] <= sc["max_suppression_pct"], f"{summary['suppression_pct']}%")
    if sc["require_benchmark"]:
        ok = bool(benchmark) and benchmark.get("level") in ("Excellent", "Good", "Moderate")
        add("Utility benchmark acceptable", ok,
            benchmark.get("level", "not run") if benchmark else "not run")

    missing = [a for a in sc["attestations"] if not approvals.get(a)]
    for a in sc["attestations"]:
        checks.append({"control": a, "status": "pass" if approvals.get(a) else "pending",
                       "auto": False, "detail": "manual attestation"})
    auto_fail = any(c["status"] == "fail" for c in checks)
    decision = "BLOCKED" if auto_fail else ("NEEDS_ATTESTATION" if missing else "READY_FOR_REVIEW")
    return {"scenario": scenario, "decision": decision, "checks": checks,
            "missing_attestations": missing, "description": sc["description"],
            "release_sha256": summary["release_sha256"]}


def build_policy_artifact(evaluation):
    body = json.dumps(evaluation, sort_keys=True, ensure_ascii=False, default=json_default)
    return {**evaluation, "artifact_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest()}

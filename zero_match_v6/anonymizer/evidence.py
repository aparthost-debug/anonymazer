# -*- coding: utf-8 -*-
"""Пакет доказательств: хеш-цепочка событий, манифест, проверка, PDF."""
import hashlib
import io
import json
import uuid
from datetime import datetime, timezone

from . import __version__, signing
from .report import json_default


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      default=json_default)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def build_audit_chain(events, run_id):
    """Хеш-цепочка: каждое событие содержит хеш предыдущего и своё время."""
    chain, prev = [], "0" * 64
    for i, (etype, payload) in enumerate(events, start=1):
        body = {"event_id": i, "run_id": run_id, "event_type": etype,
                "timestamp_utc": _now(), "payload": payload, "previous_hash": prev}
        body["event_hash"] = sha256_text(canonical_json(body))
        chain.append(body)
        prev = body["event_hash"]
    return chain


def verify_audit_chain(chain):
    prev = "0" * 64
    for i, ev in enumerate(chain, start=1):
        body = {k: v for k, v in ev.items() if k != "event_hash"}
        if ev.get("event_id") != i or ev.get("previous_hash") != prev:
            return False, f"Нарушена связность на событии {i}"
        if sha256_text(canonical_json(body)) != ev.get("event_hash"):
            return False, f"Хеш события {i} не совпадает"
        prev = ev["event_hash"]
    return True, "ok"


def build_manifest(summary, config_manifest, source_file_sha256="", policy=None,
                   benchmark=None, signer=None):
    """
    Манифест одного релиза. Содержит только хеши, метрики и конфигурацию без seed —
    ни исходных значений, ни связи «строка выгрузки -> исходная строка».
    """
    run_id = str(uuid.uuid4())
    events = [
        ("SOURCE_REGISTERED", {"file_sha256": source_file_sha256,
                               "table_fingerprint": summary["source_fingerprint"],
                               "rows": summary["rows_in"]}),
        ("PIPELINE_COMPLETED", {"status": summary["status_code"], "config_hash": summary["config_hash"],
                                "min_k": summary["min_k"], "l_min": summary["l_min"],
                                "t_max": summary["t_max"]}),
        ("RELEASE_PREPARED", {"release_csv_sha256": summary["release_sha256"],
                              "rows": summary["rows_out"], "export_blocked": summary["export_blocked"]}),
    ]
    if policy:
        events.append(("POLICY_EVALUATED", {"scenario": policy["scenario"], "decision": policy["decision"]}))
    chain = build_audit_chain(events, run_id)
    body = {
        "manifest_version": "2.0", "tool_version": __version__, "run_id": run_id,
        "created_at": _now(),
        "source": {"file_sha256": source_file_sha256, "table_fingerprint": summary["source_fingerprint"],
                   "rows": summary["rows_in"]},
        "release": {"csv_sha256": summary["release_sha256"], "rows": summary["rows_out"],
                    "export_blocked": summary["export_blocked"]},
        "pipeline": summary, "configuration": config_manifest, "policy": policy,
        "benchmark": benchmark, "audit_chain": chain,
        "assurance": {
            "tamper_evidence": "hash-chain + sha256 manifest" + (" + Ed25519" if signer else ""),
            "limitations": ["Хеш-цепочка без подписи не защищает от пересчёта цепочки тем, кто её "
                            "редактирует.",
                            "Подпись проверяйте по заранее закреплённому публичному ключу (key_id), "
                            "а не по ключу из самого манифеста.",
                            "Манифест не заменяет юридическую оценку."],
        },
    }
    body["manifest_sha256"] = sha256_text(canonical_json(body))
    if signer is not None:
        body["signature"] = {"algorithm": "Ed25519", "key_id": signer.key_id,
                             "public_key_pem": signer.public_pem,
                             "signature_b64": signer.sign(body["manifest_sha256"])}
    return body


def verify_manifest(manifest, trusted_public_pem=None):
    """
    hash_ok — манифест не менялся; chain_ok — цепочка цела; signature:
    absent | invalid | valid_untrusted (ключ взят из манифеста) | valid_trusted (совпал с закреплённым).
    """
    body = {k: v for k, v in manifest.items() if k not in ("manifest_sha256", "signature")}
    hash_ok = sha256_text(canonical_json(body)) == manifest.get("manifest_sha256")
    chain_ok, chain_msg = verify_audit_chain(manifest.get("audit_chain", []))
    sig = manifest.get("signature")
    if not sig:
        status = "absent"
    elif not signing.verify_signature(sig.get("public_key_pem"), sig.get("signature_b64"),
                                      manifest.get("manifest_sha256", "")):
        status = "invalid"
    elif trusted_public_pem and signing.key_id_from_pem(trusted_public_pem) == sig.get("key_id") \
            and trusted_public_pem.strip() == sig["public_key_pem"].strip():
        status = "valid_trusted"
    else:
        status = "valid_untrusted"
    return {"hash_ok": hash_ok, "chain_ok": chain_ok, "chain_message": chain_msg,
            "signature": status, "key_id": (sig or {}).get("key_id")}


def manifest_pdf_bytes(manifest):
    """Краткий PDF-протокол (латиница + хеши; шрифт Helvetica)."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError as e:
        raise RuntimeError("Для PDF установите пакет reportlab.") from e
    p, rel = manifest["pipeline"], manifest["release"]
    lines = [
        "Zero-Match Evidence Manifest", f"Run ID: {manifest['run_id']}",
        f"Created (UTC): {manifest['created_at']}", "",
        f"Source table fingerprint: {manifest['source']['table_fingerprint']}",
        f"Release CSV sha256:       {rel['csv_sha256']}",
        f"Rows: {manifest['source']['rows']} -> {rel['rows']}   Export blocked: {rel['export_blocked']}", "",
        f"Pipeline status: {p['status_code']}   min_k={p['min_k']} (target {p['k_target']})",
        f"l_min={p['l_min']} (target {p['l_target']})   t_max={p['t_max']} (target {p['t_target']})",
        f"Residual PII (high): {p['residual_high']}   Suppression: {p['suppression_pct']}%", "",
    ]
    if manifest.get("policy"):
        lines.append(f"Policy: {manifest['policy']['scenario']} -> {manifest['policy']['decision']}")
    lines += ["", f"Manifest sha256: {manifest['manifest_sha256']}"]
    if manifest.get("signature"):
        lines.append(f"Ed25519 key_id: {manifest['signature']['key_id']}")
    lines += ["", "Audit chain:"] + [f"  {e['event_id']}. {e['event_type']}  {e['event_hash'][:24]}..."
                                      for e in manifest["audit_chain"]]
    lines += ["", "Limitations:"] + [f"  - {t}" for t in
                                      ("Verify the signature against a pinned public key.",
                                       "Not a legal compliance opinion.")]
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = A4[1] - 40
    c.setFont("Helvetica", 9)
    for line in lines:
        c.drawString(40, y, line.encode("latin-1", "replace").decode("latin-1")[:120])
        y -= 13
        if y < 40:
            c.showPage()
            c.setFont("Helvetica", 9)
            y = A4[1] - 40
    c.save()
    return buf.getvalue()

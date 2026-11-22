# -*- coding: utf-8 -*-
"""Рабочее пространство: права, «четыре глаза», привязка согласования, журнал, гонки, блокировки."""
import os
import sqlite3
import sys
import tempfile
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anonymizer import signing, workspace as ws  # noqa: E402
from anonymizer.dossier import empty_card  # noqa: E402

ws.PBKDF2_ITERS = 1000          # быстрые тесты
PW = "Correct-horse-9battery"
SUMMARY = {"status_code": "ok", "export_blocked": False, "release_sha256": "a" * 64,
           "source_fingerprint": "b" * 64, "config_hash": "c" * 64, "storage_mode": "isolated"}


def valid_card():
    c = empty_card()
    c.update(operator_name="ООО Пример", responsible_person="Иванов И.И.", purpose="Аналитика продаж",
             legal_basis="Договор с субъектом (ст. 6 ч. 1 п. 5)", subject_categories=["Клиенты"],
             pd_categories=["Общие"], source_location="Хост A (закрытый контур)",
             result_location="Хост B (аналитика)",
             data_composition=[{"column": "Возраст", "role": "quasi_identifier", "treatment": "обобщение"}])
    c["sufficiency"].update(assessed_by="Петров П.П., DPO", assessed_on="2026-11-23",
                            conclusion="Метод достаточен", rationale="k≥5, атака связывания не выявила однозначных.")
    return c


def new_project(db, owner, signer, name="P", card=True):
    pid = ws.create_project(db, owner, name, "", signer)
    if card:
        ws.save_card(db, owner, pid, valid_card(), signer)
    return pid


def _env(with_key=True):
    d = tempfile.mkdtemp()
    db = os.path.join(d, "w.sqlite3")
    ws.init(db)
    signer = signing.load_signer(os.path.join(d, "k.pem"), create=True) if with_key else None
    token = ws.ensure_bootstrap_token(os.path.join(d, "token.txt"))
    tok = open(token, encoding="ascii").read()
    admin = ws.bootstrap_admin(db, token, tok, "admin1", PW, signer)
    return d, db, signer, admin


def _people(db, signer, admin):
    a = ws.create_user(db, admin, "analyst_a", PW, "analyst", signer)
    b = ws.create_user(db, admin, "analyst_b", PW, "analyst", signer)
    r = ws.create_user(db, admin, "reviewer_r", PW, "reviewer", signer)
    r2 = ws.create_user(db, admin, "reviewer_s", PW, "reviewer", signer)
    return a, b, r, r2


def test_bootstrap_needs_token_and_works_once():
    d = tempfile.mkdtemp()
    db = os.path.join(d, "w.sqlite3")
    ws.init(db)
    token = ws.ensure_bootstrap_token(os.path.join(d, "t.txt"))
    for bad in ("", "wrong"):
        try:
            ws.bootstrap_admin(db, token, bad, "root", PW)
            assert False
        except PermissionError:
            pass
    assert not ws.has_users(db)
    ws.bootstrap_admin(db, token, open(token, encoding="ascii").read(), "root", PW)
    assert ws.has_users(db) and not os.path.exists(token)
    try:
        ws.bootstrap_admin(db, token, "x", "root2", PW)
        assert False
    except PermissionError:
        pass


def test_password_policy_and_roles():
    d, db, signer, admin = _env()
    for bad in ("short", "alllowercase12345", "NoDigitsHereAtAll"):
        try:
            ws.create_user(db, admin, "u_" + str(len(bad)), bad, "analyst", signer)
            assert False
        except ValueError:
            pass
    a, b, r, r2 = _people(db, signer, admin)
    try:
        ws.create_user(db, a, "evil", PW, "admin", signer)        # аналитик не создаёт пользователей
        assert False
    except PermissionError:
        pass


def test_lockout_after_failed_logins_and_unknown_user():
    d, db, signer, admin = _env()
    for _ in range(ws.MAX_FAILED):
        assert ws.authenticate(db, "admin1", "wrong-password-1A", signer) is None
    try:
        ws.authenticate(db, "admin1", PW, signer)                  # даже верный пароль — заблокирован
        assert False
    except ws.AccountLocked:
        pass
    assert ws.authenticate(db, "nobody", "x" * 12, signer) is None


def test_project_acl_and_submit_rights():
    d, db, signer, admin = _env()
    a, b, r, r2 = _people(db, signer, admin)
    pid = new_project(db, a, signer, "Выгрузка")
    assert [p["id"] for p in ws.list_projects(db, a)] == [pid]
    assert ws.list_projects(db, b) == []                           # чужие проекты не видны
    assert [p["id"] for p in ws.list_projects(db, r)] == [pid]
    for actor in (b, r):                                            # не владелец / рецензент не отправляют
        try:
            ws.submit_run(db, actor, pid, SUMMARY, signer)
            assert False
        except PermissionError:
            pass
    try:
        ws.list_runs(db, b, pid)
        assert False
    except PermissionError:
        pass


def test_submit_rejects_failed_or_blocked_results():
    d, db, signer, admin = _env()
    a, b, r, r2 = _people(db, signer, admin)
    pid = new_project(db, a, signer)
    for bad in ({**SUMMARY, "status_code": "needs_work"}, {**SUMMARY, "export_blocked": True},
                {**SUMMARY, "release_sha256": "zz"}):
        try:
            ws.submit_run(db, a, pid, bad, signer)
            assert False
        except ValueError:
            pass


def test_four_eyes_and_approval_bound_to_release_hash():
    d, db, signer, admin = _env()
    a, b, r, r2 = _people(db, signer, admin)
    pid = new_project(db, a, signer)
    run1 = ws.submit_run(db, a, pid, SUMMARY, signer)
    try:
        ws.review_run(db, a, run1, "APPROVED", "сам себе", signer)  # владелец
        assert False
    except PermissionError:
        pass
    try:
        ws.review_run(db, r, run1, "REJECTED", "", signer)          # без комментария
        assert False
    except ValueError:
        pass
    # администратор, отправивший прогон, не может его же утвердить
    run_admin = ws.submit_run(db, admin, pid, {**SUMMARY, "release_sha256": "d" * 64}, signer)
    try:
        ws.review_run(db, admin, run_admin, "APPROVED", "ок", signer)
        assert False
    except PermissionError:
        pass
    # прежний прогон заменён новым -> решение по нему невозможно
    try:
        ws.review_run(db, r, run1, "APPROVED", "ок", signer)
        assert False
    except ValueError:
        pass
    ws.review_run(db, r, run_admin, "APPROVED", "проверено", signer)
    assert ws.release_status(db, pid, "d" * 64)["approved"]
    assert not ws.release_status(db, pid, "a" * 64)["approved"]      # другой файл — не утверждён
    assert not ws.release_status(db, pid, "e" * 64)["approved"]
    try:
        ws.review_run(db, r2, run_admin, "REJECTED", "поздно", signer)  # уже решён
        assert False
    except ValueError:
        pass


def test_audit_chain_survives_concurrent_writers():
    d, db, signer, admin = _env()
    a, b, r, r2 = _people(db, signer, admin)
    errs = []

    def work(i):
        try:
            for j in range(6):
                ws.create_project(db, a, f"P{i}-{j}", "", signer)
        except Exception as e:  # noqa: BLE001
            errs.append(repr(e))
    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    v = ws.verify_audit(db, signer.public_pem if signer else None)
    assert not errs and v["valid"], (errs[:1], v)
    assert v["events"] >= 36 and v["trust"] == "signed-and-verified" or not signer


def test_verify_is_read_only_and_detects_tampering():
    d, db, signer, admin = _env()
    keyfile = os.path.join(d, "k.pem")
    pem = signer.public_pem if signer else None
    os.remove(os.path.join(d, "token.txt")) if os.path.exists(os.path.join(d, "token.txt")) else None
    if signer:
        os.remove(keyfile)
        v = ws.verify_audit(db, pem)                                  # ключа нет, проверка идёт по публичному
        assert v["valid"] and not os.path.exists(keyfile)             # verify НЕ создаёт ключ
    con = sqlite3.connect(db)
    con.execute("UPDATE workspace_audit SET payload_json='{\"x\":1}' WHERE id=1")
    con.commit()
    con.close()
    bad = ws.verify_audit(db, pem)
    assert not bad["valid"] and "1" in bad["reason"]


def test_unsigned_log_is_reported_as_hash_chain_only_and_wrong_key_fails():
    d, db, _, admin = _env(with_key=False)
    v = ws.verify_audit(db)
    assert v["valid"] and v["trust"] == "hash-chain-only" and v["signed_events"] == 0
    if signing.crypto_available():
        d2, db2, signer2, _ = _env()
        other = signing.load_signer(os.path.join(d2, "other.pem"), create=True)
        assert not ws.verify_audit(db2, other.public_pem)["valid"]


def test_change_password_and_last_admin_protected():
    d, db, signer, admin = _env()
    a, b, r, r2 = _people(db, signer, admin)
    try:
        ws.change_password(db, a, "wrong", "New-Password-12X", signer)
        assert False
    except PermissionError:
        pass
    ws.change_password(db, a, PW, "New-Password-12X", signer)
    assert ws.authenticate(db, "analyst_a", "New-Password-12X", signer)["id"] == a
    assert ws.authenticate(db, "analyst_a", PW, signer) is None
    try:
        ws.set_user_active(db, admin, admin, False, signer)
        assert False
    except PermissionError:
        pass
    ws.set_user_active(db, admin, b, False, signer)
    assert ws.authenticate(db, "analyst_b", PW, signer) is None


def test_duplicate_username_is_a_clean_error():
    d, db, signer, admin = _env()
    ws.create_user(db, admin, "same_name", PW, "analyst", signer)
    try:
        ws.create_user(db, admin, "same_name", PW, "reviewer", signer)
        assert False
    except ValueError as e:
        assert "существует" in str(e)
    assert ws.verify_audit(db, signer.public_pem if signer else None)["valid"]   # откат не ломает журнал


def test_partially_signed_log_is_reported_honestly():
    if not signing.crypto_available():
        return
    d = tempfile.mkdtemp()
    db = os.path.join(d, "w.sqlite3")
    ws.init(db)
    token = ws.ensure_bootstrap_token(os.path.join(d, "t.txt"))
    admin = ws.bootstrap_admin(db, token, open(token, encoding="ascii").read(), "root", PW, None)  # без ключа
    signer = signing.load_signer(os.path.join(d, "k.pem"), create=True)
    ws.create_user(db, admin, "later_user", PW, "analyst", signer)
    v = ws.verify_audit(db, signer.public_pem)
    assert v["valid"] and v["trust"] == "partially-signed" and v["verified_signatures"] == 1


def test_submit_requires_isolation_and_complete_card():
    d, db, signer, admin = _env()
    a, b, r, r2 = _people(db, signer, admin)
    no_card = new_project(db, a, signer, "без карточки", card=False)
    try:
        ws.submit_run(db, a, no_card, SUMMARY, signer)
        assert False
    except ValueError as e:
        assert "карточки" in str(e)
    pid = ws.create_project(db, a, "неполная", "", signer)
    ws.save_card(db, a, pid, empty_card(), signer)
    try:
        ws.submit_run(db, a, pid, SUMMARY, signer)
        assert False
    except ValueError as e:
        assert "не заполнена" in str(e)
    ok = new_project(db, a, signer, "полная")
    try:
        ws.submit_run(db, a, ok, {**SUMMARY, "storage_mode": "shared (dev)"}, signer)
        assert False
    except ValueError as e:
        assert "не изолирован" in str(e)
    assert ws.submit_run(db, a, ok, SUMMARY, signer)


def test_card_acl_snapshot_and_audit():
    d, db, signer, admin = _env()
    a, b, r, r2 = _people(db, signer, admin)
    pid = new_project(db, a, signer)
    for actor in (b, r):                                        # чужой аналитик и рецензент не редактируют
        try:
            ws.save_card(db, actor, pid, valid_card(), signer)
            assert False
        except PermissionError:
            pass
    try:
        ws.get_card(db, b, pid)                                 # и не читают чужую карточку
        assert False
    except PermissionError:
        pass
    assert ws.get_card(db, r, pid)["card"]["purpose"] == "Аналитика продаж"      # рецензент читает
    run = ws.submit_run(db, a, pid, SUMMARY, signer)
    snap = ws.list_runs(db, r, pid)[0]["summary"]
    assert snap["card"]["operator_name"] == "ООО Пример" and snap["card_sha256"] == ws.get_card(db, a, pid)["card_sha256"]
    changed = valid_card()
    changed["purpose"] = "Совсем другая цель"
    new_sha = ws.save_card(db, a, pid, changed, signer)                         # правка после отправки
    assert new_sha != snap["card_sha256"]
    assert ws.list_runs(db, r, pid)[0]["summary"]["card"]["purpose"] == "Аналитика продаж"   # снимок неизменен
    assert run and ws.verify_audit(db, signer.public_pem if signer else None)["valid"]
    con = sqlite3.connect(db)
    actions = [x[0] for x in con.execute("SELECT action FROM workspace_audit")]
    con.close()
    assert actions.count("CARD_UPDATED") >= 2 and "RUN_SUBMITTED" in actions

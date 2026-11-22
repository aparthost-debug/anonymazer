# -*- coding: utf-8 -*-
"""
Рабочее пространство: пользователи, проекты, согласование, подписанный журнал аудита (SQLite).

Принципы:
  * права проверяются здесь, а не в интерфейсе; изменение состояния и запись аудита — в одной
    транзакции (BEGIN IMMEDIATE), поэтому цепочка не «раздваивается» при параллельной работе;
  * согласование привязано к конкретному прогону, sha256 выгружаемого CSV и снимку карточки обезличивания;
  * на согласование принимаются только прогоны в изолированном режиме и при заполненной карточке
    (Приказ РКН № 140, Прил. 1, п. 1(2), 1(3), 1(6));
  * «четыре глаза»: утверждающий ≠ владелец проекта ≠ тот, кто отправил прогон;
  * проверка журнала только читает; ключ подписи создаётся лишь явно при init(create_key=True).
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import signing
from .dossier import card_hash, normalize_card, validate_card
from .report import json_default

ROLES = ("admin", "analyst", "reviewer")
PBKDF2_ITERS = int(os.environ.get("ZERO_MATCH_PBKDF2_ITERS", "600000"))
MAX_FAILED, LOCK_MINUTES, MIN_PASSWORD = 5, 15, 12
_DUMMY_SALT = b"zero-match-dummy-salt"


class AccountLocked(Exception):
    pass


# ---------- инфраструктура ----------
def default_paths():
    home = signing.home_dir()
    return {"db": str(home / "workspace.sqlite3"), "key": str(home / "audit_ed25519.pem"),
            "token": str(home / "bootstrap_token.txt")}


def default_signer():
    """Загружает существующий ключ (не создаёт). None — подписи не используются."""
    return signing.load_signer(default_paths()["key"], create=False)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canon(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      default=json_default)


@contextmanager
def _tx(db):
    con = sqlite3.connect(db, timeout=30, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    try:
        con.execute("BEGIN IMMEDIATE")
        yield con
        con.execute("COMMIT")
    except BaseException:
        try:
            con.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        con.close()


@contextmanager
def _ro(db):
    con = sqlite3.connect(db, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, salt BLOB NOT NULL,
  pw_hash BLOB NOT NULL, iters INTEGER NOT NULL, role TEXT NOT NULL
  CHECK(role IN ('admin','analyst','reviewer')), active INTEGER NOT NULL DEFAULT 1,
  failed_count INTEGER NOT NULL DEFAULT 0, locked_until TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS projects(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, purpose TEXT NOT NULL DEFAULT '',
  owner_id INTEGER NOT NULL REFERENCES users(id), status TEXT NOT NULL DEFAULT 'DRAFT',
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS project_cards(
  project_id INTEGER PRIMARY KEY REFERENCES projects(id), card_json TEXT NOT NULL,
  card_sha256 TEXT NOT NULL, updated_by INTEGER NOT NULL REFERENCES users(id), updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER NOT NULL REFERENCES projects(id),
  actor_id INTEGER NOT NULL REFERENCES users(id), release_sha256 TEXT NOT NULL,
  source_fingerprint TEXT NOT NULL, status TEXT NOT NULL, summary_json TEXT NOT NULL,
  created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS approvals(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL REFERENCES runs(id),
  project_id INTEGER NOT NULL REFERENCES projects(id),
  reviewer_id INTEGER NOT NULL REFERENCES users(id), decision TEXT NOT NULL,
  comment TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS workspace_audit(
  id INTEGER PRIMARY KEY, actor_id INTEGER, action TEXT NOT NULL, object_type TEXT NOT NULL,
  object_id TEXT NOT NULL, payload_json TEXT NOT NULL, timestamp_utc TEXT NOT NULL,
  previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL, key_id TEXT, signature_b64 TEXT);
"""


def init(db, create_key=False):
    """Создаёт схему. create_key=True — явно создаёт ключ подписи (если cryptography доступна)."""
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db, isolation_level=None)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(SCHEMA)
    finally:
        con.close()
    if create_key:
        return signing.load_signer(default_paths()["key"], create=True)
    return None


def _audit(con, signer, actor_id, action, otype, oid, payload):
    """Добавляет событие в цепочку. Вызывать ТОЛЬКО внутри _tx."""
    last = con.execute("SELECT id, event_hash FROM workspace_audit ORDER BY id DESC LIMIT 1").fetchone()
    eid, prev = (last["id"] + 1, last["event_hash"]) if last else (1, "0" * 64)
    body = {"id": eid, "actor_id": actor_id, "action": action, "object_type": otype,
            "object_id": str(oid), "payload": payload, "timestamp_utc": _now(),
            "previous_hash": prev, "key_id": signer.key_id if signer else None}
    digest = hashlib.sha256(_canon(body).encode("utf-8")).hexdigest()
    sig = signer.sign(digest) if signer else None
    con.execute(
        "INSERT INTO workspace_audit(id,actor_id,action,object_type,object_id,payload_json,"
        "timestamp_utc,previous_hash,event_hash,key_id,signature_b64) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (eid, actor_id, action, str(otype), str(oid), _canon(payload), body["timestamp_utc"], prev,
         digest, body["key_id"], sig))


# ---------- пароли и пользователи ----------
def _hash_pw(password, salt, iters):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)


def _check_new_password(pw):
    if len(pw) < MIN_PASSWORD:
        raise ValueError(f"Пароль короче {MIN_PASSWORD} символов.")
    if pw.lower() == pw or not re.search(r"\d", pw):
        raise ValueError("Пароль должен содержать заглавную букву и цифру.")


def _insert_user(con, signer, actor_id, username, password, role):
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,64}", username):
        raise ValueError("Логин: 3–64 символа, латиница, цифры, _ . -")
    if role not in ROLES:
        raise ValueError("Неизвестная роль.")
    _check_new_password(password)
    salt = secrets.token_bytes(16)
    try:
        cur = con.execute(
            "INSERT INTO users(username,salt,pw_hash,iters,role,created_at) VALUES(?,?,?,?,?,?)",
            (username, salt, _hash_pw(password, salt, PBKDF2_ITERS), PBKDF2_ITERS, role, _now()))
    except sqlite3.IntegrityError as e:
        raise ValueError("Такой логин уже существует.") from e
    _audit(con, signer, actor_id, "USER_CREATED", "user", cur.lastrowid,
           {"username": username, "role": role})
    return cur.lastrowid


def _user(con, user_id):
    r = con.execute("SELECT * FROM users WHERE id=? AND active=1", (user_id,)).fetchone()
    if not r:
        raise PermissionError("Пользователь не найден или отключён.")
    return r


def has_users(db):
    with _ro(db) as con:
        return con.execute("SELECT COUNT(*) FROM users").fetchone()[0] > 0


def ensure_bootstrap_token(token_path):
    """Токен первичной настройки лежит на сервере: первый посетитель сайта админом не станет."""
    p = Path(token_path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(secrets.token_urlsafe(24), encoding="ascii")
        try:
            p.chmod(0o600)
        except OSError:
            pass
    return str(p)


def bootstrap_admin(db, token_path, token, username, password, signer=None):
    with _tx(db) as con:
        if con.execute("SELECT COUNT(*) FROM users").fetchone()[0] > 0:
            raise PermissionError("Пользователи уже созданы.")
        expected = Path(token_path).read_text(encoding="ascii").strip() if Path(token_path).exists() else ""
        if not expected or not hmac.compare_digest(token.strip(), expected):
            raise PermissionError("Неверный токен первичной настройки.")
        uid = _insert_user(con, signer, None, username, password, "admin")
    try:
        Path(token_path).unlink()
    except OSError:
        pass
    return uid


def create_user(db, actor_id, username, password, role, signer=None):
    with _tx(db) as con:
        if _user(con, actor_id)["role"] != "admin":
            raise PermissionError("Создавать пользователей может только admin.")
        return _insert_user(con, signer, actor_id, username, password, role)


def authenticate(db, username, password, signer=None):
    """Пользователь или None. Блокировка после MAX_FAILED ошибок; время ответа не выдаёт, есть ли логин."""
    with _tx(db) as con:
        row = con.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if row is None or not row["active"]:
            _hash_pw(password, _DUMMY_SALT, PBKDF2_ITERS)
            return None
        if row["locked_until"] and row["locked_until"] > _now():
            raise AccountLocked(f"Учётная запись заблокирована до {row['locked_until']} UTC.")
        ok = hmac.compare_digest(_hash_pw(password, row["salt"], row["iters"]), row["pw_hash"])
        if ok:
            con.execute("UPDATE users SET failed_count=0, locked_until=NULL WHERE id=?", (row["id"],))
            return {"id": row["id"], "username": row["username"], "role": row["role"]}
        failed = row["failed_count"] + 1
        until = None
        if failed >= MAX_FAILED:
            until = (datetime.now(timezone.utc) + timedelta(minutes=LOCK_MINUTES)).isoformat(timespec="seconds")
            failed = 0
            _audit(con, signer, row["id"], "ACCOUNT_LOCKED", "user", row["id"], {"minutes": LOCK_MINUTES})
        con.execute("UPDATE users SET failed_count=?, locked_until=? WHERE id=?", (failed, until, row["id"]))
        return None


def change_password(db, user_id, old, new, signer=None):
    with _tx(db) as con:
        u = _user(con, user_id)
        if not hmac.compare_digest(_hash_pw(old, u["salt"], u["iters"]), u["pw_hash"]):
            raise PermissionError("Текущий пароль неверен.")
        _check_new_password(new)
        salt = secrets.token_bytes(16)
        con.execute("UPDATE users SET salt=?, pw_hash=?, iters=? WHERE id=?",
                    (salt, _hash_pw(new, salt, PBKDF2_ITERS), PBKDF2_ITERS, user_id))
        _audit(con, signer, user_id, "PASSWORD_CHANGED", "user", user_id, {})


def set_user_active(db, actor_id, user_id, active, signer=None):
    with _tx(db) as con:
        if _user(con, actor_id)["role"] != "admin":
            raise PermissionError("Только admin.")
        if not active:
            target = con.execute("SELECT role FROM users WHERE id=?", (user_id,)).fetchone()
            admins = con.execute("SELECT COUNT(*) FROM users WHERE role='admin' AND active=1").fetchone()[0]
            if target and target["role"] == "admin" and admins <= 1:
                raise PermissionError("Нельзя отключить последнего администратора.")
        con.execute("UPDATE users SET active=? WHERE id=?", (1 if active else 0, user_id))
        _audit(con, signer, actor_id, "USER_ACTIVE_SET", "user", user_id, {"active": bool(active)})


def list_users(db, actor_id):
    with _ro(db) as con:
        if con.execute("SELECT role FROM users WHERE id=? AND active=1", (actor_id,)).fetchone()["role"] != "admin":
            raise PermissionError("Только admin.")
        return [dict(r) for r in con.execute(
            "SELECT id, username, role, active, created_at FROM users ORDER BY id")]


# ---------- проекты и согласование ----------
def create_project(db, actor_id, name, purpose="", signer=None):
    if not name.strip():
        raise ValueError("Название проекта пустое.")
    with _tx(db) as con:
        if _user(con, actor_id)["role"] not in ("admin", "analyst"):
            raise PermissionError("Проекты создают analyst и admin.")
        now = _now()
        cur = con.execute("INSERT INTO projects(name,purpose,owner_id,created_at,updated_at) VALUES(?,?,?,?,?)",
                          (name.strip(), purpose.strip(), actor_id, now, now))
        _audit(con, signer, actor_id, "PROJECT_CREATED", "project", cur.lastrowid, {"name": name.strip()})
        return cur.lastrowid


def _can_see(user, project):
    return user["role"] in ("admin", "reviewer") or project["owner_id"] == user["id"]


def list_projects(db, actor_id):
    with _ro(db) as con:
        u = con.execute("SELECT id, role FROM users WHERE id=? AND active=1", (actor_id,)).fetchone()
        if not u:
            raise PermissionError("Пользователь не найден.")
        rows = con.execute("SELECT p.*, o.username AS owner FROM projects p JOIN users o ON o.id=p.owner_id "
                           "ORDER BY p.id DESC").fetchall()
        return [dict(r) for r in rows if _can_see(u, r)]


def save_card(db, actor_id, project_id, card, signer=None):
    """Сохраняет карточку обезличивания проекта (владелец или admin). Каждое изменение — в журнале."""
    card = normalize_card(card)
    sha = card_hash(card)
    with _tx(db) as con:
        u = _user(con, actor_id)
        proj = con.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        if not proj:
            raise ValueError("Проект не найден.")
        if u["role"] not in ("admin", "analyst") or (u["role"] != "admin" and proj["owner_id"] != actor_id):
            raise PermissionError("Карточку редактирует владелец проекта или admin.")
        con.execute("INSERT INTO project_cards(project_id,card_json,card_sha256,updated_by,updated_at) VALUES(?,?,?,?,?) "
                    "ON CONFLICT(project_id) DO UPDATE SET card_json=excluded.card_json, "
                    "card_sha256=excluded.card_sha256, updated_by=excluded.updated_by, updated_at=excluded.updated_at",
                    (project_id, _canon(card), sha, actor_id, _now()))
        _audit(con, signer, actor_id, "CARD_UPDATED", "project", project_id,
               {"card_sha256": sha, "open_issues": len(validate_card(card))})
    return sha


def get_card(db, actor_id, project_id):
    with _ro(db) as con:
        u = con.execute("SELECT id, role FROM users WHERE id=? AND active=1", (actor_id,)).fetchone()
        proj = con.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        if not u or not proj or not _can_see(u, proj):
            raise PermissionError("Нет доступа к проекту.")
        row = con.execute("SELECT * FROM project_cards WHERE project_id=?", (project_id,)).fetchone()
        return None if not row else {"card": json.loads(row["card_json"]), "card_sha256": row["card_sha256"],
                                     "updated_at": row["updated_at"]}


def submit_run(db, actor_id, project_id, summary, signer=None):
    """Отправляет на согласование КОНКРЕТНЫЙ результат: sha256 выгрузки и сводку проверок."""
    sha = str(summary.get("release_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("В сводке нет корректного release_sha256.")
    if summary.get("status_code") not in ("ok", "ok_warn") or summary.get("export_blocked"):
        raise ValueError("Результат не прошёл обязательные проверки — на согласование не отправляется.")
    if summary.get("storage_mode") != "isolated":
        raise ValueError("Режим хранения не изолирован: исходные и обезличенные данные не должны храниться "
                         "совместно (Приказ № 140, Прил. 1, п. 1(6)). Запустите в изолированном режиме.")
    with _tx(db) as con:
        u = _user(con, actor_id)
        proj = con.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        if not proj:
            raise ValueError("Проект не найден.")
        if u["role"] not in ("admin", "analyst") or (u["role"] != "admin" and proj["owner_id"] != actor_id):
            raise PermissionError("Отправлять прогоны может владелец проекта или admin.")
        row = con.execute("SELECT card_json, card_sha256 FROM project_cards WHERE project_id=?", (project_id,)).fetchone()
        if not row:
            raise ValueError("У проекта нет карточки обезличивания — заполните и сохраните её.")
        card = json.loads(row["card_json"])
        issues = validate_card(card)
        if issues:
            raise ValueError("Карточка обезличивания не заполнена: " + "; ".join(issues[:5]))
        snapshot = {**summary, "card": card, "card_sha256": row["card_sha256"]}      # неизменяемый снимок
        con.execute("UPDATE runs SET status='SUPERSEDED' WHERE project_id=? AND status='SUBMITTED'", (project_id,))
        cur = con.execute(
            "INSERT INTO runs(project_id,actor_id,release_sha256,source_fingerprint,status,summary_json,created_at) "
            "VALUES(?,?,?,?,'SUBMITTED',?,?)",
            (project_id, actor_id, sha, summary.get("source_fingerprint", ""), _canon(snapshot), _now()))
        con.execute("UPDATE projects SET status='PENDING_REVIEW', updated_at=? WHERE id=?", (_now(), project_id))
        _audit(con, signer, actor_id, "RUN_SUBMITTED", "run", cur.lastrowid,
               {"project_id": project_id, "release_sha256": sha, "config_hash": summary.get("config_hash"),
                "card_sha256": row["card_sha256"], "storage_mode": summary.get("storage_mode")})
        return cur.lastrowid


def review_run(db, reviewer_id, run_id, decision, comment="", signer=None):
    if decision not in ("APPROVED", "REJECTED"):
        raise ValueError("Решение: APPROVED или REJECTED.")
    if decision == "REJECTED" and not comment.strip():
        raise ValueError("Для отклонения нужен комментарий.")
    with _tx(db) as con:
        u = _user(con, reviewer_id)
        if u["role"] not in ("admin", "reviewer"):
            raise PermissionError("Согласовывать могут reviewer и admin.")
        run = con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not run or run["status"] != "SUBMITTED":
            raise ValueError("Прогон не ожидает решения (уже решён или заменён новым).")
        proj = con.execute("SELECT * FROM projects WHERE id=?", (run["project_id"],)).fetchone()
        if reviewer_id in (proj["owner_id"], run["actor_id"]):
            raise PermissionError("Принцип «четырёх глаз»: утверждающий не может быть владельцем "
                                  "проекта или автором отправки.")
        con.execute("INSERT INTO approvals(run_id,project_id,reviewer_id,decision,comment,created_at) "
                    "VALUES(?,?,?,?,?,?)", (run_id, proj["id"], reviewer_id, decision, comment.strip(), _now()))
        con.execute("UPDATE runs SET status=? WHERE id=?", (decision, run_id))
        con.execute("UPDATE projects SET status=?, updated_at=? WHERE id=?",
                    ("APPROVED" if decision == "APPROVED" else "REJECTED", _now(), proj["id"]))
        _audit(con, signer, reviewer_id, f"RUN_{decision}", "run", run_id,
               {"project_id": proj["id"], "release_sha256": run["release_sha256"], "comment": comment.strip()})


def list_runs(db, actor_id, project_id):
    with _ro(db) as con:
        u = con.execute("SELECT id, role FROM users WHERE id=? AND active=1", (actor_id,)).fetchone()
        proj = con.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        if not u or not proj or not _can_see(u, proj):
            raise PermissionError("Нет доступа к проекту.")
        out = []
        for r in con.execute("SELECT r.*, a.username AS actor FROM runs r JOIN users a ON a.id=r.actor_id "
                             "WHERE project_id=? ORDER BY r.id DESC", (project_id,)):
            d = dict(r)
            d["summary"] = json.loads(d.pop("summary_json"))
            out.append(d)
        return out


def release_status(db, project_id, release_sha256):
    """True, только если ИМЕННО этот файл (sha256) утверждён в этом проекте."""
    with _ro(db) as con:
        row = con.execute("SELECT id FROM runs WHERE project_id=? AND release_sha256=? AND status='APPROVED' "
                          "LIMIT 1", (project_id, release_sha256)).fetchone()
        return {"approved": row is not None, "run_id": row["id"] if row else None}


# ---------- проверка и экспорт журнала (только чтение) ----------
def verify_audit(db, public_pem=None):
    """
    Проверяет цепочку и (если дан публичный ключ) подписи. Ничего не создаёт и не меняет.
    trust: signed-and-verified | partially-signed | hash-chain-only (подписей нет или ключ не передан).
    """
    with _ro(db) as con:
        events = [dict(r) for r in con.execute("SELECT * FROM workspace_audit ORDER BY id")]
    prev, signed, verified = "0" * 64, 0, 0
    key_id = signing.key_id_from_pem(public_pem) if public_pem else None
    for i, ev in enumerate(events, start=1):
        body = {"id": ev["id"], "actor_id": ev["actor_id"], "action": ev["action"],
                "object_type": ev["object_type"], "object_id": ev["object_id"],
                "payload": json.loads(ev["payload_json"]), "timestamp_utc": ev["timestamp_utc"],
                "previous_hash": ev["previous_hash"], "key_id": ev["key_id"]}
        if ev["id"] != i or ev["previous_hash"] != prev:
            return {"valid": False, "events": len(events), "reason": f"Нарушена связность на событии {i}",
                    "trust": "broken", "signed_events": signed, "verified_signatures": verified}
        if hashlib.sha256(_canon(body).encode("utf-8")).hexdigest() != ev["event_hash"]:
            return {"valid": False, "events": len(events), "reason": f"Хеш события {i} не совпадает",
                    "trust": "broken", "signed_events": signed, "verified_signatures": verified}
        if ev["signature_b64"]:
            signed += 1
            if public_pem:
                if ev["key_id"] != key_id or not signing.verify_signature(public_pem, ev["signature_b64"],
                                                                           ev["event_hash"]):
                    return {"valid": False, "events": len(events), "reason": f"Подпись события {i} неверна",
                            "trust": "broken", "signed_events": signed, "verified_signatures": verified}
                verified += 1
        prev = ev["event_hash"]
    if events and verified == len(events):
        trust = "signed-and-verified"
    elif verified:
        trust = "partially-signed"      # часть событий создана до появления ключа
    else:
        trust = "hash-chain-only"
    return {"valid": True, "events": len(events), "reason": "ok", "trust": trust,
            "signed_events": signed, "verified_signatures": verified}


def export_audit_bundle(db, public_pem=None):
    with _ro(db) as con:
        events = [dict(r) for r in con.execute("SELECT * FROM workspace_audit ORDER BY id")]
    return {"verification": verify_audit(db, public_pem), "public_key_pem": public_pem,
            "key_id": signing.key_id_from_pem(public_pem) if public_pem else None, "events": events}

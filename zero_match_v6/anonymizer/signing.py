# -*- coding: utf-8 -*-
"""Подпись Ed25519 (необязательная, нужен пакет cryptography). Ключ создаётся ТОЛЬКО явно."""
import base64
import hashlib
import os
from pathlib import Path


def home_dir():
    """Каталог данных рабочего пространства (БД, ключи, токен первичной настройки)."""
    d = Path(os.environ.get("ZERO_MATCH_HOME", "zero_match_data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def key_id_from_pem(public_pem):
    return hashlib.sha256(public_pem.strip().encode("ascii")).hexdigest()[:16]


class Signer:
    def __init__(self, private_key, public_pem):
        self._key = private_key
        self.public_pem = public_pem
        self.key_id = key_id_from_pem(public_pem)

    def sign(self, digest_hex):
        return base64.b64encode(self._key.sign(bytes.fromhex(digest_hex))).decode("ascii")


def crypto_available():
    try:
        import cryptography  # noqa: F401
        return True
    except ImportError:
        return False


def load_signer(private_path, create=False):
    """
    Загружает приватный ключ. create=True — создаёт пару (приватный 0600 + *.pub.pem).
    Возвращает None, если cryptography не установлена или ключа нет (и create=False).
    """
    if not crypto_available():
        return None
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    path = Path(private_path)
    if path.exists():
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    elif create:
        key = Ed25519PrivateKey.generate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                           serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
        try:
            path.chmod(0o600)
        except OSError:
            pass
    else:
        return None
    pem = key.public_key().public_bytes(serialization.Encoding.PEM,
                                        serialization.PublicFormat.SubjectPublicKeyInfo).decode("ascii")
    pub_path = Path(str(path) + ".pub.pem")
    if not pub_path.exists():
        pub_path.write_text(pem, encoding="ascii")
    return Signer(key, pem)


def verify_signature(public_pem, signature_b64, digest_hex):
    """Только проверка: ключи не создаёт и не меняет."""
    if not crypto_available() or not public_pem or not signature_b64:
        return False
    try:
        from cryptography.hazmat.primitives import serialization
        pub = serialization.load_pem_public_key(public_pem.encode("ascii"))
        pub.verify(base64.b64decode(signature_b64), bytes.fromhex(digest_hex))
        return True
    except Exception:
        return False

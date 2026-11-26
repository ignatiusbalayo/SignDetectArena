"""Encryption helpers shared by participants and the scoring bot.

Two schemes are used:

* Submissions (hybrid): a random AES-256-GCM key encrypts the file and is itself
  wrapped with the competition RSA public key (RSA-OAEP / SHA-256). Only the
  scoring bot, which holds the private key as a repository secret, can decrypt.
* Private test labels (symmetric): AES-256-GCM with a 32-byte key stored as the
  repository secret ``LABELS_KEY`` (base64). The encrypted labels file is safe
  to commit; the key is not.

File formats
------------
submission: b"ARENAS1" | u16 wrapped_key_len | wrapped_key | 12B nonce | ciphertext
labels:     b"ARENAL1" | 12B nonce | ciphertext
"""

from __future__ import annotations

import base64
import os
import struct

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

SUB_MAGIC = b"ARENAS1"
LAB_MAGIC = b"ARENAL1"
NONCE_LEN = 12


def _oaep() -> padding.OAEP:
    return padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None,
    )


# ----------------------------------------------------------------- key helpers
def generate_rsa_keypair() -> tuple[bytes, bytes]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    private_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_pem = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def generate_labels_key() -> str:
    return base64.b64encode(AESGCM.generate_key(bit_length=256)).decode()


# ----------------------------------------------------------------- submissions
def encrypt_submission(plaintext: bytes, public_pem: bytes) -> bytes:
    public_key = serialization.load_pem_public_key(public_pem)
    session_key = AESGCM.generate_key(bit_length=256)
    nonce = os.urandom(NONCE_LEN)
    ciphertext = AESGCM(session_key).encrypt(nonce, plaintext, SUB_MAGIC)
    wrapped = public_key.encrypt(session_key, _oaep())
    return SUB_MAGIC + struct.pack(">H", len(wrapped)) + wrapped + nonce + ciphertext


def decrypt_submission(blob: bytes, private_pem: bytes) -> bytes:
    if not blob.startswith(SUB_MAGIC):
        raise ValueError("Not an encrypted submission (bad header).")
    pos = len(SUB_MAGIC)
    (wlen,) = struct.unpack(">H", blob[pos : pos + 2])
    pos += 2
    wrapped = blob[pos : pos + wlen]
    pos += wlen
    nonce = blob[pos : pos + NONCE_LEN]
    ciphertext = blob[pos + NONCE_LEN :]
    private_key = serialization.load_pem_private_key(private_pem, password=None)
    session_key = private_key.decrypt(wrapped, _oaep())
    return AESGCM(session_key).decrypt(nonce, ciphertext, SUB_MAGIC)


# ---------------------------------------------------------------------- labels
def encrypt_labels(plaintext: bytes, key_b64: str) -> bytes:
    key = base64.b64decode(key_b64)
    nonce = os.urandom(NONCE_LEN)
    return LAB_MAGIC + nonce + AESGCM(key).encrypt(nonce, plaintext, LAB_MAGIC)


def decrypt_labels(blob: bytes, key_b64: str) -> bytes:
    if not blob.startswith(LAB_MAGIC):
        raise ValueError("Not an encrypted labels file (bad header).")
    key = base64.b64decode(key_b64)
    nonce = blob[len(LAB_MAGIC) : len(LAB_MAGIC) + NONCE_LEN]
    return AESGCM(key).decrypt(nonce, blob[len(LAB_MAGIC) + NONCE_LEN :], LAB_MAGIC)

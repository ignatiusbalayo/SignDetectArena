"""ORGANIZER ONLY - create the competition encryption keys (run once).

Writes:
  encryption/public_key.pem    commit this (participants encrypt with it)
  encryption/private_key.pem   NEVER commit (git-ignored) -> GitHub secret SUBMISSION_PRIVATE_KEY
  encryption/labels.key        NEVER commit (git-ignored) -> GitHub secret LABELS_KEY

Usage:
  python scripts/generate_keys.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "competition"))
from crypto_utils import generate_labels_key, generate_rsa_keypair  # noqa: E402


def main() -> None:
    enc = ROOT / "encryption"
    enc.mkdir(exist_ok=True)
    if (enc / "private_key.pem").exists():
        sys.exit("encryption/private_key.pem already exists - refusing to overwrite.")

    private_pem, public_pem = generate_rsa_keypair()
    (enc / "private_key.pem").write_bytes(private_pem)
    (enc / "public_key.pem").write_bytes(public_pem)
    (enc / "labels.key").write_text(generate_labels_key())

    print("Keys written to encryption/.\n")
    print("Next steps:")
    print("  1. Commit encryption/public_key.pem (public).")
    print("  2. In GitHub -> Settings -> Secrets and variables -> Actions add:")
    print("       SUBMISSION_PRIVATE_KEY = full contents of encryption/private_key.pem")
    print("       LABELS_KEY             = contents of encryption/labels.key")
    print("  3. Back both secret files up somewhere private (password manager).")
    print("  4. Run: python scripts/encrypt_labels.py")


if __name__ == "__main__":
    main()

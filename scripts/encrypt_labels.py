"""ORGANIZER ONLY - encrypt the private test labels so they can live in the repo.

Reads  data/private/test_labels.csv   (git-ignored, plaintext)
Writes encryption/test_labels.enc     (safe to commit)

The key is read from the LABELS_KEY environment variable, or from
encryption/labels.key if the variable is not set.

Usage:
  python scripts/encrypt_labels.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "competition"))
from crypto_utils import encrypt_labels  # noqa: E402


def main() -> None:
    src = ROOT / "data" / "private" / "test_labels.csv"
    dst = ROOT / "encryption" / "test_labels.enc"
    if not src.exists():
        sys.exit(f"Missing {src}. Run scripts/prepare_data.py first.")

    key = os.environ.get("LABELS_KEY")
    if not key:
        key_file = ROOT / "encryption" / "labels.key"
        if not key_file.exists():
            sys.exit("No LABELS_KEY env var and no encryption/labels.key. Run scripts/generate_keys.py.")
        key = key_file.read_text().strip()

    dst.parent.mkdir(exist_ok=True)
    dst.write_bytes(encrypt_labels(src.read_bytes(), key))
    print(f"Encrypted labels -> {dst} ({dst.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

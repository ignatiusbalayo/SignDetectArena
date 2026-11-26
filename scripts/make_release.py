"""ORGANIZER ONLY - zip data/public (+ sample files) for a GitHub Release.

Usage:  python scripts/make_release.py
Output: SignDetectArena_Dataset.zip   (git-ignored; upload it to a GitHub Release, e.g. v1.0)
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "SignDetectArena_Dataset.zip"


def main() -> None:
    public = ROOT / "data" / "public"
    if not public.exists():
        raise SystemExit("data/public not found. Run scripts/prepare_data.py first.")
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(public.rglob("*")):
            if path.is_file():
                zf.write(path, Path("SignDetectArena_Dataset") / path.relative_to(ROOT))
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

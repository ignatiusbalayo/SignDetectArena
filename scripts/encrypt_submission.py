"""PARTICIPANTS - encrypt your predictions and prepare a submission folder.

Usage:
  python scripts/encrypt_submission.py predictions.csv --team my_team \
      --model "Faster R-CNN" --type human --notes "short free text"

--type is one of: human, llm, hybrid  (the benchmark compares humans and LLMs).

Creates:
  submissions/inbox/<team>/submission.enc   your encrypted predictions
  submissions/inbox/<team>/metadata.json    team / model / type / notes

Then commit ONLY that folder and open a Pull Request to main.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "competition"))
from crypto_utils import encrypt_submission  # noqa: E402

TEAM_RE = re.compile(r"^[A-Za-z0-9_-]{2,40}$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", help="Path to your predictions.csv")
    ap.add_argument("--team", required=True, help="Unique team name: letters, digits, _ or - (2-40 chars)")
    ap.add_argument("--model", required=True, help="Model name, e.g. 'U-Net' or 'GPT-5 agent'")
    ap.add_argument("--type", default="human", choices=["human", "llm", "hybrid"])
    ap.add_argument("--notes", default="", help="Optional short note (max 140 chars)")
    args = ap.parse_args()

    if not TEAM_RE.match(args.team):
        sys.exit("Team name must match [A-Za-z0-9_-] and be 2-40 characters long.")

    pred = Path(args.predictions)
    if not pred.exists():
        sys.exit(f"File not found: {pred}")

    pub = ROOT / "encryption" / "public_key.pem"
    if not pub.exists():
        sys.exit("encryption/public_key.pem not found. Pull the latest repository version.")

    out_dir = ROOT / "submissions" / "inbox" / args.team
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "submission.enc").write_bytes(encrypt_submission(pred.read_bytes(), pub.read_bytes()))
    meta = {
        "team": args.team,
        "model": args.model[:60],
        "type": args.type,
        "notes": args.notes[:140],
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"Created {out_dir}/submission.enc and metadata.json")
    print("Next: git add submissions/inbox/%s && git commit && open a Pull Request." % args.team)


if __name__ == "__main__":
    main()

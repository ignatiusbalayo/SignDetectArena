"""Render leaderboard/leaderboard.csv -> leaderboard.md and docs/leaderboard.csv.

Kaggle-style ranking: higher score is better, ties share the same rank.
"""

import csv
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "leaderboard" / "leaderboard.csv"
MD_PATH = ROOT / "leaderboard" / "leaderboard.md"
DOCS_CSV = ROOT / "docs" / "leaderboard.csv"


def main() -> None:
    cfg = yaml.safe_load((ROOT / "competition" / "config.yaml").read_text())
    rows = []
    if CSV_PATH.exists():
        with CSV_PATH.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: float(r["score"]), reverse=True)

    lines = [
        f"# {cfg['competition_name']} Leaderboard",
        "",
        f"Ranked by **{cfg['primary_metric']}** (higher is better). Ties share a rank.",
        "",
        f"| Rank | Team | Model | Type | {cfg['primary_metric']} | {cfg['secondary_metric']} | Submitted (UTC) |",
        "|-----:|:-----|:------|:-----|------:|------:|:----------------|",
    ]
    rank, prev = 0, None
    for i, r in enumerate(rows, start=1):
        if r["score"] != prev:
            rank, prev = i, r["score"]
        cells = [str(rank), r["team"], r["model"], r["type"], r["score"], r["secondary"], r["timestamp_utc"]]
        lines.append("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    if not rows:
        lines.append("| - | _no submissions yet_ | | | | | |")

    MD_PATH.parent.mkdir(exist_ok=True)
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if CSV_PATH.exists():
        DOCS_CSV.parent.mkdir(exist_ok=True)
        shutil.copyfile(CSV_PATH, DOCS_CSV)
    print(f"Rendered {len(rows)} entries.")


if __name__ == "__main__":
    main()

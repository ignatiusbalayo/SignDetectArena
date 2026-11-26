"""CI orchestrator for scoring a submission Pull Request.

Runs from the *trusted* base branch inside GitHub Actions. Only the participant's
data files (submissions/inbox/<team>/{submission.enc,metadata.json}) are copied
from the PR, and never executed.

Sub-commands
------------
verify  <changed_files.txt>
    Checks that a PR only touches submissions/inbox/<team>/{submission.enc,metadata.json}
    and exactly one team. Prints the team name on success.

score   --team T --pr N --result result.json
    Decrypts the submission and the private labels, validates, evaluates, enforces
    the per-team submission limit and appends a row to leaderboard/leaderboard.csv.
    Always writes result.json (status ok|error) for the PR comment step.

Environment (score): SUBMISSION_PRIVATE_KEY, LABELS_KEY
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from crypto_utils import decrypt_labels, decrypt_submission  # noqa: E402

TEAM_RE = re.compile(r"^[A-Za-z0-9_-]{2,40}$")
ALLOWED_FILES = {"submission.enc", "metadata.json"}
MAX_BYTES = 20 * 1024 * 1024
LEADERBOARD = ROOT / "leaderboard" / "leaderboard.csv"
FIELDS = ["team", "model", "type", "score", "secondary", "timestamp_utc", "notes", "pr"]


def load_config() -> dict:
    return yaml.safe_load((HERE / "config.yaml").read_text())


def safe_cell(value: object) -> str:
    """One-line text, protected against spreadsheet formula injection."""
    text = " ".join(str(value).split())[:140]
    return "'" + text if text[:1] in "=+-@" else text


# --------------------------------------------------------------------- verify
def cmd_verify(changed_file: str) -> int:
    paths = [p.strip() for p in Path(changed_file).read_text().splitlines() if p.strip()]
    if not paths:
        print("No changed files found.", file=sys.stderr)
        return 1
    teams, names = set(), set()
    for p in paths:
        parts = p.split("/")
        ok = (
            len(parts) == 4
            and parts[0] == "submissions"
            and parts[1] == "inbox"
            and TEAM_RE.match(parts[2])
            and parts[3] in ALLOWED_FILES
        )
        if not ok:
            print(
                f"Rejected: '{p}'. A submission PR may only add "
                "submissions/inbox/<team>/submission.enc and metadata.json.",
                file=sys.stderr,
            )
            return 1
        teams.add(parts[2])
        names.add(parts[3])
    if len(teams) != 1:
        print("A PR must contain exactly one team folder.", file=sys.stderr)
        return 1
    if "submission.enc" not in names:
        print("submission.enc is missing from the PR.", file=sys.stderr)
        return 1
    print(next(iter(teams)))
    return 0


# ---------------------------------------------------------------------- score
def _count_existing(team: str) -> int:
    if not LEADERBOARD.exists():
        return 0
    with LEADERBOARD.open(newline="") as fh:
        return sum(1 for r in csv.DictReader(fh) if r["team"].lower() == team.lower())


def _fail(result_path: Path, message: str, **extra) -> int:
    result_path.write_text(json.dumps({"status": "error", "message": message, **extra}, indent=2))
    print(f"ERROR: {message}", file=sys.stderr)
    return 1


def cmd_score(team: str, pr: str, result_path: Path) -> int:
    import evaluate as ev
    import validate_submission as vs

    cfg = load_config()
    if not TEAM_RE.match(team):
        return _fail(result_path, "Invalid team name.")

    folder = ROOT / "submissions" / "inbox" / team
    sub_path = folder / "submission.enc"
    if not sub_path.is_file():
        return _fail(result_path, "submission.enc not found.", team=team)
    if sub_path.stat().st_size > MAX_BYTES:
        return _fail(result_path, f"Submission larger than {MAX_BYTES // 2**20} MB.", team=team)

    meta = {"model": "Unknown", "type": "unknown", "notes": ""}
    meta_path = folder / "metadata.json"
    if meta_path.is_file() and meta_path.stat().st_size < 10_000:
        try:
            meta.update({k: v for k, v in json.loads(meta_path.read_text()).items() if k in meta})
        except Exception:
            pass

    limit = int(cfg.get("max_submissions_per_team", 1))
    if _count_existing(team) >= limit:
        return _fail(
            result_path,
            f"Team '{team}' already used its {limit} allowed submission(s).",
            team=team,
        )

    priv = os.environ.get("SUBMISSION_PRIVATE_KEY", "")
    key = os.environ.get("LABELS_KEY", "")
    labels_enc = ROOT / "encryption" / "test_labels.enc"
    if not priv or not key or not labels_enc.is_file():
        return _fail(result_path, "Scoring bot is not configured (missing secrets or encrypted labels).")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        try:
            (tmp_dir / "pred.csv").write_bytes(
                decrypt_submission(sub_path.read_bytes(), priv.encode())
            )
        except Exception:
            return _fail(
                result_path,
                "Could not decrypt submission. Encrypt it with scripts/encrypt_submission.py "
                "using the repository's current encryption/public_key.pem.",
                team=team,
            )
        try:
            (tmp_dir / "labels.csv").write_bytes(decrypt_labels(labels_enc.read_bytes(), key))
        except Exception:
            return _fail(result_path, "Scoring bot could not decrypt the private labels.")

        try:
            vs.validate(tmp_dir / "pred.csv", ROOT / "data" / "public" / cfg.get("test_file", "test_images.csv"))
        except ValueError as exc:
            return _fail(result_path, f"Validation failed: {exc}", team=team)

        try:
            metrics = ev.evaluate(tmp_dir / "pred.csv", tmp_dir / "labels.csv")
        except Exception:
            return _fail(result_path, "Evaluation failed on this file (unexpected content).", team=team)

    primary, secondary = cfg["primary_metric"], cfg["secondary_metric"]
    LEADERBOARD.parent.mkdir(exist_ok=True)
    new_file = not LEADERBOARD.exists()
    with LEADERBOARD.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "team": team,
                "model": safe_cell(meta["model"]),
                "type": safe_cell(meta["type"]),
                "score": f"{metrics[primary]:.4f}",
                "secondary": f"{metrics[secondary]:.4f}",
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "notes": safe_cell(meta["notes"]),
                "pr": pr,
            }
        )

    result_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "team": team,
                "model": safe_cell(meta["model"]),
                "primary_name": primary,
                "primary": round(metrics[primary], 4),
                "secondary_name": secondary,
                "secondary": round(metrics[secondary], 4),
            },
            indent=2,
        )
    )
    print(f"{team}: {primary}={metrics[primary]:.4f} {secondary}={metrics[secondary]:.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("changed_file")
    s = sub.add_parser("score")
    s.add_argument("--team", required=True)
    s.add_argument("--pr", default="")
    s.add_argument("--result", default="result.json")
    args = ap.parse_args()
    if args.cmd == "verify":
        return cmd_verify(args.changed_file)
    return cmd_score(args.team, args.pr, Path(args.result))


if __name__ == "__main__":
    sys.exit(main())

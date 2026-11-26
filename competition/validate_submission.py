"""Validate a node-prediction submission.

CLI:   python competition/validate_submission.py predictions.csv
Format (CSV with header, one row per (node, class) you want to score):

    id,class_id,score

* id        a node id from data/public/test_nodes.csv (column `id`; also listed in test_ids.csv)
* class_id  integer in [0, 10)
* score     float in [0, 1], your confidence that this node's box is an object of that class
* each (id, class_id) pair at most once; a node may appear with several classes
* a header-only file is valid (scores 0)

The scorer gives every row the box of its node, removes duplicates with per-class NMS and
keeps the 100 best detections per image (per image and class only the
300 best-scoring nodes enter NMS).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent


def _find_test_csv(cfg) -> Path:
    name = cfg.get("test_file", "test_ids.csv")
    for p in [Path("data/public") / name, HERE.parent / "data" / "public" / name,
              Path("SignDetectArena_Dataset/data/public") / name]:
        if p.exists():
            return p
    raise ValueError(f"Could not find data/public/{name}.")


def validate(pred_path, test_csv=None) -> bool:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    test_csv = Path(test_csv) if test_csv else _find_test_csv(cfg)
    try:
        preds = pd.read_csv(pred_path)
    except Exception as exc:
        raise ValueError(f"Could not read prediction file: {exc}")
    if list(preds.columns) != ["id", "class_id", "score"]:
        raise ValueError("Columns must be exactly: id,class_id,score")
    if preds.empty:
        return True
    ids = set(pd.read_csv(test_csv, usecols=["id"])["id"])
    for col in ("id", "class_id", "score"):
        v = pd.to_numeric(preds[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(v).all():
            raise ValueError(f"Column {col} must contain only finite numbers.")
        preds[col] = v
    if not set(preds["id"].astype(int)).issubset(ids) or (preds["id"] % 1 != 0).any():
        raise ValueError("Some ids are not valid test node ids.")
    if (preds["class_id"] % 1 != 0).any() or (preds["class_id"] < 0).any() or (preds["class_id"] >= cfg["num_classes"]).any():
        raise ValueError(f"class_id must be an integer in [0, {cfg['num_classes']}).")
    if (preds["score"] < 0).any() or (preds["score"] > 1).any():
        raise ValueError("score must be in [0, 1].")
    if preds.duplicated(["id", "class_id"]).any():
        raise ValueError("Each (id, class_id) pair may appear only once.")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_submission.py <pred_path>", file=sys.stderr)
        sys.exit(1)
    try:
        validate(sys.argv[1])
        print("VALID SUBMISSION")
    except Exception as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

"""Score a node-prediction submission.

CLI (prints only the primary score):
    python competition/evaluate.py <predictions.csv> <test_labels.csv>

The scoring bot imports evaluate() instead.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.append(os.path.dirname(__file__))
from metrics import calculate_metrics, detections_from_nodes  # noqa: E402

HERE = Path(__file__).resolve().parent


def evaluate(pred_path, label_path) -> dict:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    preds = pd.read_csv(pred_path)
    labels = pd.read_csv(label_path, dtype={"image_id": str})
    gt = labels[labels["kind"] == "gt"].astype({"class_id": int})
    dets = detections_from_nodes(preds, labels, float(cfg["nms_iou"]), int(cfg["max_detections_per_image"]),
                                 float(cfg["min_score"]), int(cfg.get("max_candidates_per_class", 300)))
    return calculate_metrics(dets.astype({"class_id": int}), gt)


def main(pred_path, label_path) -> None:
    try:
        scores = evaluate(pred_path, label_path)
    except Exception as exc:  # keep stdout clean: only the score goes there
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"{scores['map50']:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <pred_path> <label_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

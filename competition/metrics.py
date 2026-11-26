"""COCO-style detection metrics implemented with numpy only (plus node-to-detection conversion).

AP is computed per class with 101-point interpolated precision, at IoU thresholds
0.50, 0.55, ..., 0.95. Classes that have no ground-truth box in the test set are
ignored. Greedy matching: detections are processed in descending score order and
each is matched to the best still-unmatched ground-truth box of the same class and
image whose IoU is >= the threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)
BOX_COLS = ["x_min", "y_min", "x_max", "y_max"]


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (N,4) and b (M,4) in xyxy format."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def _ap_from_pr(tp_flags: np.ndarray, n_gt: int) -> float:
    """101-point interpolated AP from a score-sorted array of TP flags."""
    if n_gt == 0:
        return float("nan")
    if len(tp_flags) == 0:
        return 0.0
    tp = np.cumsum(tp_flags)
    fp = np.cumsum(1 - tp_flags)
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-12)
    # Make precision monotonically non-increasing from the right.
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, grid, side="left")
    sampled = np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0)
    return float(sampled.mean())


def _class_ap(dets: pd.DataFrame, gts: pd.DataFrame, thr: float) -> float:
    n_gt = len(gts)
    if n_gt == 0:
        return float("nan")
    gt_by_img = {k: g[BOX_COLS].to_numpy(float) for k, g in gts.groupby("image_id")}
    used = {k: np.zeros(len(v), dtype=bool) for k, v in gt_by_img.items()}
    dets = dets.sort_values("score", ascending=False, kind="mergesort")
    flags = np.zeros(len(dets))
    for i, (img, box) in enumerate(zip(dets["image_id"].to_numpy(), dets[BOX_COLS].to_numpy(float))):
        g = gt_by_img.get(img)
        if g is None:
            continue
        ious = box_iou(box[None, :], g)[0]
        ious[used[img]] = -1.0
        j = int(np.argmax(ious))
        if ious[j] >= thr:
            used[img][j] = True
            flags[i] = 1.0
    return _ap_from_pr(flags, n_gt)


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> np.ndarray:
    """Indices kept by greedy non-maximum suppression."""
    order = np.argsort(-scores, kind="mergesort")
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        order = rest[box_iou(boxes[i][None, :], boxes[rest])[0] <= thr]
    return np.array(keep, dtype=int)


def detections_from_nodes(preds: pd.DataFrame, labels: pd.DataFrame, nms_iou: float = 0.5,
                          max_det: int = 100, min_score: float = 0.001, max_candidates: int = 300) -> pd.DataFrame:
    """Turn node predictions (id,class_id,score) into detections.

    labels holds one `kind=node` row per test node (id, image_id and the node's box). Each prediction
    inherits its node's box; per image and class only the `max_candidates` best scores enter NMS,
    which removes duplicates; the best `max_det` detections per image are kept.
    """
    nodes = labels[labels["kind"] == "node"][["id", "image_id"] + BOX_COLS].copy()
    nodes["id"] = nodes["id"].astype(int)
    p = preds[["id", "class_id", "score"]].copy()
    p["id"] = p["id"].astype(int)
    p = p[p["score"] >= min_score].merge(nodes, on="id", how="inner")
    out = []
    for (img, c), g in p.groupby(["image_id", "class_id"]):
        g = g.nlargest(max_candidates, "score")
        k = nms(g[BOX_COLS].to_numpy(float), g["score"].to_numpy(float), nms_iou)
        out.append(g.iloc[k])
    if not out:
        return pd.DataFrame(columns=["image_id", "class_id", "score"] + BOX_COLS)
    d = pd.concat(out).sort_values(["image_id", "score"], ascending=[True, False], kind="mergesort")
    d = d.groupby("image_id", sort=False).head(max_det)
    return d[["image_id", "class_id", "score"] + BOX_COLS].reset_index(drop=True)


def calculate_metrics(preds: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """preds: image_id,class_id,score,x_min..y_max ; labels: image_id,class_id,x_min..y_max."""
    classes = sorted(labels["class_id"].unique())
    ap_by_thr = []
    for thr in IOU_THRESHOLDS:
        aps = []
        for c in classes:
            ap = _class_ap(preds[preds["class_id"] == c], labels[labels["class_id"] == c], float(thr))
            if not np.isnan(ap):
                aps.append(ap)
        ap_by_thr.append(float(np.mean(aps)) if aps else 0.0)
    return {"map50": ap_by_thr[0], "map50_95": float(np.mean(ap_by_thr))}

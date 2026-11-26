"""ORGANIZER ONLY - turn the staged images into the public GRAPH dataset.

Input (private, from scripts/prepare_data.py):
    data/private/staging/{train,test}/images/*.jpg, {train,test}_index.csv, {train,test}_boxes.csv

Each image becomes ONE graph of candidate boxes ("proposals"):
  * proposals = bounding boxes of regions found by hierarchical merging of SLIC superpixels
                (Selective-Search style: several superpixel granularities x three merge
                strategies, every region in every hierarchy is a proposal, near-duplicates removed)
  * nodes     = proposals, with 71 hand-crafted features (box geometry, colour, texture, shape)
  * edges     = overlapping / nested proposals (box IoU or containment >= --edge-thr), top-K per node
  * node label = class of the ground-truth box with the highest IoU if that IoU >= 0.5, else -1
                 (background); best_iou is also given for training

NO raw pixels are published. Participants only receive these graphs.

Outputs (public, committed):
    data/public/train_nodes.csv     image_id,node_id,<71 features>,label,best_iou
    data/public/train_edges.csv     image_id,source,target,iou
    data/public/train_boxes.csv     image_id,class_id,x_min,y_min,x_max,y_max   (ground truth)
    data/public/test_nodes.csv      id,image_id,node_id,<71 features>            (no labels)
    data/public/test_ids.csv        id   (tiny file the scoring bot validates against)
    data/public/test_edges.csv      image_id,source,target,iou
    data/public/classes.csv         class_id,name
    data/public/sample_submission.csv   id,class_id,score
    data/public/dataset_stats.json
Output (private, git-ignored):
    data/private/test_labels.csv    kind,id,image_id,class_id,x_min,y_min,x_max,y_max
        kind=gt   : a ground-truth box (id = -1)
        kind=node : the box of test node `id` (class_id = -1)
Then run: python scripts/encrypt_labels.py

Usage:
    python scripts/build_graphs.py
    python scripts/build_graphs.py --quick            # test split only: proposal count and recall, writes nothing
    python scripts/build_graphs.py --levels 25,80,250 --dedup 0.75
Needs: pip install scikit-image
"""

import argparse
import heapq
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    from skimage.color import rgb2gray, rgb2hsv, rgb2lab
    from skimage.feature import local_binary_pattern
    from skimage.filters import laplace, sobel, sobel_h, sobel_v
    from skimage.segmentation import slic
except ImportError:
    sys.exit("scikit-image is required: pip install scikit-image")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "competition"))
from metrics import box_iou, calculate_metrics, detections_from_nodes  # noqa: E402

STAGE = ROOT / "data" / "private" / "staging"
PUB = ROOT / "data" / "public"
PRIV = ROOT / "data" / "private"
BOX = ["x_min", "y_min", "x_max", "y_max"]
FEATURES = BOX + [
    "cx", "cy", "w", "h", "log_aspect", "area_frac", "fill_ratio", "img_w", "img_h",
    "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std",
    "hue_cos", "hue_sin", "sat_mean", "val_mean", "grad_mean", "grad_std", "lap_var",
    "ring_contrast", "sat_std", "n_pixels_frac", "n_parts",
]
EXTRA = ([f"hue_h{i}" for i in range(8)] + [f"sat_h{i}" for i in range(4)] + [f"val_h{i}" for i in range(4)]
         + [f"ori{i}" for i in range(8)] + [f"lbp{i}" for i in range(10)]
         + ["edge_density", "hist_chi2", "d_r", "d_g", "d_b", "d_grad", "dist_img"])
FEATURES = FEATURES + EXTRA
MIN_SIDE, MAX_AREA_FRAC = 8, 0.80
STRATEGIES = ("all", "color", "size")
DEFAULT_LEVELS = "20,60,200,500:100,1200:56"
SIZE_EDGES, SIZE_QUOTA = (24.0, 48.0, 96.0), (0.35, 0.35, 0.20, 0.10)
SWEEP = ["20,60,200,500:100", "20,60,200,500:100,1200:56", "20,60,200,500:100,1200:56,2500:40"]


# ------------------------------------------------------------------ proposals
def merge_regions(seg: np.ndarray, lab: np.ndarray, strategy: str = "all", cap: float = 0):
    """Hierarchical grouping of superpixels. Returns list of (x1, y1, x2, y2, n_pixels, n_parts).

    cap > 0: two regions are never merged if the box around them would be longer than `cap` pixels
    (used for the fine levels, which only look for small objects).
    """
    h, w = seg.shape
    n = int(seg.max()) + 1
    flat = seg.ravel()
    cnt = np.bincount(flat, minlength=n).astype(float)
    col = np.stack([np.bincount(flat, weights=lab[..., k].ravel(), minlength=n) for k in range(3)], 1)
    yy, xx = np.mgrid[0:h, 0:w]
    x1 = np.full(n, w, float); x2 = np.zeros(n); y1 = np.full(n, h, float); y2 = np.zeros(n)
    np.minimum.at(x1, flat, xx.ravel()); np.maximum.at(x2, flat, xx.ravel() + 1.0)
    np.minimum.at(y1, flat, yy.ravel()); np.maximum.at(y2, flat, yy.ravel() + 1.0)

    pairs = np.concatenate([np.stack([seg[:, :-1].ravel(), seg[:, 1:].ravel()], 1),
                            np.stack([seg[:-1, :].ravel(), seg[1:, :].ravel()], 1)])
    pairs = np.unique(np.sort(pairs[pairs[:, 0] != pairs[:, 1]], axis=1), axis=0)
    nbrs = {i: set() for i in range(n)}
    for a_, b_ in pairs:
        nbrs[int(a_)].add(int(b_)); nbrs[int(b_)].add(int(a_))

    reg = {i: dict(c=float(cnt[i]), col=tuple(map(float, col[i])), b=[float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])], parts=1)
           for i in range(n)}
    out = [(*reg[i]["b"], reg[i]["c"], 1) for i in range(n)]
    area_img = float(h * w)

    def union_box(ra, rb):
        return [min(ra["b"][0], rb["b"][0]), min(ra["b"][1], rb["b"][1]),
                max(ra["b"][2], rb["b"][2]), max(ra["b"][3], rb["b"][3])]

    def sim(a, b):
        ra, rb = reg[a], reg[b]
        bx = union_box(ra, rb)
        if cap and max(bx[2] - bx[0], bx[3] - bx[1]) > cap:
            return None
        ca, cb = ra["col"], rb["col"]
        ia, ib = 1.0 / ra["c"], 1.0 / rb["c"]
        dc = math.sqrt((ca[0] * ia - cb[0] * ib) ** 2 + (ca[1] * ia - cb[1] * ib) ** 2 + (ca[2] * ia - cb[2] * ib) ** 2) / 100.0
        fill = ((bx[2] - bx[0]) * (bx[3] - bx[1]) - ra["c"] - rb["c"]) / area_img
        size = (ra["c"] + rb["c"]) / area_img
        if strategy == "color":
            return dc + 0.5 * size
        if strategy == "size":
            return 3.0 * size + fill
        return dc + 2.0 * size + fill

    heap = []
    for a_ in nbrs:
        for b_ in nbrs[a_]:
            if a_ < b_:
                v = sim(a_, b_)
                if v is not None:
                    heap.append((v, a_, b_))
    heapq.heapify(heap)
    alive = set(range(n))
    nxt = n
    while heap:
        _, a_, b_ = heapq.heappop(heap)
        if a_ not in alive or b_ not in alive:
            continue
        ra, rb = reg[a_], reg[b_]
        reg[nxt] = dict(c=ra["c"] + rb["c"], col=(ra["col"][0] + rb["col"][0], ra["col"][1] + rb["col"][1], ra["col"][2] + rb["col"][2]), b=union_box(ra, rb), parts=ra["parts"] + rb["parts"])
        alive -= {a_, b_}
        alive.add(nxt)
        nb = (nbrs[a_] | nbrs[b_]) - {a_, b_}
        nbrs[nxt] = set(nb)
        for x in nb:
            nbrs[x] -= {a_, b_}
            nbrs[x].add(nxt)
            v = sim(x, nxt)
            if v is not None:
                heapq.heappush(heap, (v, min(x, nxt), max(x, nxt)))
        out.append((*reg[nxt]["b"], reg[nxt]["c"], reg[nxt]["parts"]))
        nxt += 1
    return out


def parse_levels(spec: str):
    """'25,80,600:96' -> [(25, 0), (80, 0), (600, 96)]  (N superpixels, optional box-size cap in px)."""
    out = []
    for item in spec.split(","):
        n, _, c = item.partition(":")
        out.append((int(n), float(c) if c else 0.0))
    return out


def proposals_for(img: np.ndarray, levels, compactness: float, dedup: float):
    h, w, _ = img.shape
    lab = rgb2lab(img)
    allp = []
    for ns, cap in levels:
        seg = slic(img, n_segments=ns, compactness=compactness, sigma=1.0, start_label=0, channel_axis=-1)
        _, seg = np.unique(seg, return_inverse=True)
        seg = seg.reshape(h, w)
        for strat in STRATEGIES:
            allp += merge_regions(seg, lab, strat, cap)
    arr = np.array(allp, dtype=float)
    keep = ((arr[:, 2] - arr[:, 0]) >= MIN_SIDE) & ((arr[:, 3] - arr[:, 1]) >= MIN_SIDE) & \
           ((arr[:, 2] - arr[:, 0]) * (arr[:, 3] - arr[:, 1]) <= MAX_AREA_FRAC * h * w)
    arr = arr[keep]
    # remove near-duplicate boxes (keep the one with more parts: it is the more informative region)
    arr = arr[np.argsort(-arr[:, 5], kind="mergesort")]
    dropped = np.zeros(len(arr), bool)
    for i in range(len(arr) - 1):
        if not dropped[i]:
            dropped[i + 1:] |= box_iou(arr[i:i + 1, :4], arr[i + 1:, :4])[0] > dedup
    arr = arr[~dropped]
    return arr[np.lexsort((arr[:, 0], arr[:, 1]))]  # stable, position-based node order


# ------------------------------------------------------------------ features
def integral(a: np.ndarray) -> np.ndarray:
    return np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def box_sum(I: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = (b[:, 0].astype(int), b[:, 1].astype(int), np.ceil(b[:, 2]).astype(int), np.ceil(b[:, 3]).astype(int))
    return I[y2, x2] - I[y1, x2] - I[y2, x1] + I[y1, x1]


def extra_features(img, hsv, gray, b, ex, area, ring_area, ms, ring) -> dict:
    """Histogram / texture / contrast descriptors for every box (all computed with integral images)."""
    h, w, _ = img.shape
    colourful = (hsv[..., 1] >= 0.2) & (hsv[..., 2] >= 0.2)
    hue_bin = np.minimum((hsv[..., 0] * 8).astype(int), 7)
    sat_bin = np.minimum((hsv[..., 1] * 4).astype(int), 3)
    val_bin = np.minimum((hsv[..., 2] * 4).astype(int), 3)
    gy, gx = sobel_h(gray), sobel_v(gray)
    mag = np.hypot(gx, gy)
    ori_bin = np.minimum((np.arctan2(gy, gx) % np.pi) / np.pi * 8, 7.999).astype(int)
    lbp = local_binary_pattern((gray * 255).astype(np.uint8), 8, 1, "uniform").astype(int)

    def box_and_ring(mask):
        I = integral(mask.astype(np.float32)).astype(np.float64)
        inner = box_sum(I, b)
        return inner, box_sum(I, ex) - inner

    out, colour_hist_box, colour_hist_ring = {}, [], []
    for name, masks, norm_by_mag in (
        ("hue_h", [(hue_bin == k) & colourful for k in range(8)], False),
        ("sat_h", [sat_bin == k for k in range(4)], False),
        ("val_h", [val_bin == k for k in range(4)], False),
    ):
        for k, m in enumerate(masks):
            inner, ring_ = box_and_ring(m)
            out[f"{name}{k}"] = inner / area
            colour_hist_box.append(inner / area)
            colour_hist_ring.append(ring_ / ring_area)
    mag_in, mag_ring = box_and_ring(mag)
    for k in range(8):
        inner, _ = box_and_ring(mag * (ori_bin == k))
        out[f"ori{k}"] = inner / np.maximum(mag_in, 1e-6)
    for k in range(10):
        inner, _ = box_and_ring(lbp == k)
        out[f"lbp{k}"] = inner / area
    edge_in, _ = box_and_ring(mag > 0.1)
    out["edge_density"] = edge_in / area
    hb, hr = np.stack(colour_hist_box, 1), np.stack(colour_hist_ring, 1)
    out["hist_chi2"] = ((hb - hr) ** 2 / (hb + hr + 1e-6)).sum(1)
    for i, c in enumerate("rgb"):
        out[f"d_{c}"] = ms[i][0] - ring[i]
    out["d_grad"] = mag_in / area - mag_ring / ring_area
    mean_img = img.reshape(-1, 3).mean(0)
    out["dist_img"] = np.sqrt(sum((ms[i][0] - mean_img[i]) ** 2 for i in range(3)))
    return out


def node_features(img: np.ndarray, props: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    hsv, gray = rgb2hsv(img), rgb2gray(img)
    grad, lap = sobel(gray), laplace(gray)
    chans = [img[..., 0], img[..., 1], img[..., 2], np.cos(2 * np.pi * hsv[..., 0]),
             np.sin(2 * np.pi * hsv[..., 0]), hsv[..., 1], hsv[..., 2], grad, lap]
    ints = [integral(c) for c in chans]
    ints2 = [integral(c * c) for c in chans]
    b = props[:, :4].copy()
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, w)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, h)
    bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    area = bw * bh

    def mean_std(k, bb, ar):
        m = box_sum(ints[k], bb) / ar
        v = box_sum(ints2[k], bb) / ar - m * m
        return m, np.sqrt(np.clip(v, 0, None))

    ms = [mean_std(k, b, area) for k in range(len(chans))]
    # ring contrast: colour distance between the box and the ring around it (box expanded by 25%)
    ex = np.stack([np.clip(b[:, 0] - 0.25 * bw, 0, w), np.clip(b[:, 1] - 0.25 * bh, 0, h),
                   np.clip(b[:, 2] + 0.25 * bw, 0, w), np.clip(b[:, 3] + 0.25 * bh, 0, h)], 1)
    ea = np.maximum((ex[:, 2] - ex[:, 0]) * (ex[:, 3] - ex[:, 1]), 1.0)
    ring_area = np.maximum(ea - area, 1.0)
    ring = [(box_sum(ints[k], ex) - box_sum(ints[k], b)) / ring_area for k in range(3)]
    contrast = np.sqrt(sum((ms[k][0] - ring[k]) ** 2 for k in range(3)))
    extra = extra_features(img, hsv, gray, b, ex, area, ring_area, ms, ring)
    f = {
        "x_min": b[:, 0], "y_min": b[:, 1], "x_max": b[:, 2], "y_max": b[:, 3],
        "cx": (b[:, 0] + b[:, 2]) / 2 / w, "cy": (b[:, 1] + b[:, 3]) / 2 / h, "w": bw / w, "h": bh / h,
        "log_aspect": np.log(bw / bh), "area_frac": area / (w * h), "fill_ratio": props[:, 4] / area,
        "img_w": np.full(len(b), w, float), "img_h": np.full(len(b), h, float),
        "r_mean": ms[0][0], "g_mean": ms[1][0], "b_mean": ms[2][0],
        "r_std": ms[0][1], "g_std": ms[1][1], "b_std": ms[2][1],
        "hue_cos": ms[3][0], "hue_sin": ms[4][0], "sat_mean": ms[5][0], "val_mean": ms[6][0],
        "grad_mean": ms[7][0], "grad_std": ms[7][1], "lap_var": ms[8][1] ** 2,
        "ring_contrast": contrast, "sat_std": ms[5][1],
        "n_pixels_frac": props[:, 4] / (w * h), "n_parts": props[:, 5],
    }
    f.update(extra)
    return np.stack([f[k] for k in FEATURES], 1)


def select_top(feats: np.ndarray, max_nodes: int) -> np.ndarray:
    """Indices (sorted) of at most max_nodes proposals, chosen WITHOUT labels.

    Objectness = contrast against the surrounding ring x (0.5 + fill ratio). Boxes are split into four
    size classes with fixed quotas (35% / 35% / 20% / 10% for <24, 24-48, 48-96, >=96 px) so small
    objects are not crowded out; unused quota is handed to the best remaining boxes.
    """
    n = len(feats)
    if not max_nodes or n <= max_nodes:
        return np.arange(n)
    ix = {k: FEATURES.index(k) for k in ("x_min", "y_min", "x_max", "y_max", "ring_contrast", "fill_ratio")}
    side = np.sqrt((feats[:, ix["x_max"]] - feats[:, ix["x_min"]]) * (feats[:, ix["y_max"]] - feats[:, ix["y_min"]]))
    score = feats[:, ix["ring_contrast"]] * (0.5 + feats[:, ix["fill_ratio"]])
    cls = np.digitize(side, SIZE_EDGES)
    chosen = np.zeros(n, bool)
    for c, q in enumerate(SIZE_QUOTA):
        members = np.flatnonzero(cls == c)
        chosen[members[np.argsort(-score[members], kind="mergesort")[: int(round(q * max_nodes))]]] = True
    short = max_nodes - int(chosen.sum())
    if short > 0:
        rest = np.flatnonzero(~chosen)
        chosen[rest[np.argsort(-score[rest], kind="mergesort")[:short]]] = True
    return np.flatnonzero(chosen)


def image_job(path: str, levels, compactness: float, dedup: float, max_nodes: int, with_edges: bool, edge_thr: float, edge_topk: int):
    """Everything that needs pixels for ONE image (runs in a worker process)."""
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    props = proposals_for(img, levels, compactness, dedup)
    n_raw = len(props)
    feats = node_features(img, props)
    feats = feats[select_top(feats, max_nodes)]
    pr, iv = graph_edges(feats[:, :4], edge_thr, edge_topk) if with_edges else (None, None)
    return feats, pr, iv, n_raw


def run_jobs(paths, args, with_edges: bool):
    jobs = [(str(p), args.levels, args.compactness, args.dedup, args.max_nodes, with_edges, args.edge_thr, args.edge_topk) for p in paths]
    if args.workers <= 1:
        for j in jobs:
            yield image_job(*j)
        return
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        yield from ex.map(image_job, *zip(*jobs), chunksize=2)


def graph_edges(boxes: np.ndarray, thr: float, topk: int):
    iou = box_iou(boxes, boxes)
    a1, a2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), None
    ix = np.clip(np.minimum(boxes[:, None, 2], boxes[None, :, 2]) - np.maximum(boxes[:, None, 0], boxes[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(boxes[:, None, 3], boxes[None, :, 3]) - np.maximum(boxes[:, None, 1], boxes[None, :, 1]), 0, None)
    contain = ix * iy / np.minimum(a1[:, None], a1[None, :])
    score = np.maximum(iou, 0.9 * contain)
    np.fill_diagonal(score, -1)
    pairs = set()
    for i in range(len(boxes)):
        nb = np.argsort(-score[i])[:topk]
        for j in nb:
            if score[i, j] >= thr:
                pairs.add((min(i, int(j)), max(i, int(j))))
    pairs = np.array(sorted(pairs), dtype=int).reshape(-1, 2)
    return pairs, iou[pairs[:, 0], pairs[:, 1]] if len(pairs) else np.zeros(0)


def label_nodes(boxes: np.ndarray, gt: pd.DataFrame):
    if len(gt) == 0:
        return np.full(len(boxes), -1), np.zeros(len(boxes))
    iou = box_iou(boxes, gt[BOX].to_numpy(float))
    j = iou.argmax(1)
    best = iou[np.arange(len(boxes)), j]
    cls = gt["class_id"].to_numpy()[j]
    return np.where(best >= 0.5, cls, -1), best


def build_split(split: str, args):
    index = pd.read_csv(STAGE / f"{split}_index.csv", dtype={"image_id": str})
    gts = pd.read_csv(STAGE / f"{split}_boxes.csv", dtype={"image_id": str})
    gt_by = {k: g for k, g in gts.groupby("image_id")}
    paths = [STAGE / split / "images" / f for f in index["file_name"]]
    nodes, edges = [], []
    for k, (iid, (feats, pr, iv, _)) in enumerate(zip(index["image_id"], run_jobs(paths, args, True)), 1):
        label, best = label_nodes(feats[:, :4], gt_by.get(iid, gts.iloc[:0]))
        df = pd.DataFrame(np.round(feats, 4), columns=FEATURES)
        df.insert(0, "node_id", np.arange(len(df)))
        df.insert(0, "image_id", iid)
        df["label"], df["best_iou"] = label, np.round(best, 4)
        nodes.append(df)
        edges.append(pd.DataFrame({"image_id": iid, "source": pr[:, 0], "target": pr[:, 1], "iou": np.round(iv, 4)}))
        if k % 50 == 0:
            print(f"  {split}: {k}/{len(index)} graphs", flush=True)
    return pd.concat(nodes, ignore_index=True), pd.concat(edges, ignore_index=True), gts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=parse_levels, default=parse_levels(DEFAULT_LEVELS),
                    help="superpixel counts, optionally with a box-size cap in px, e.g. 25,80,250,600:96")
    ap.add_argument("--limit", type=int, default=0, help="--quick only: use the first N test images")
    ap.add_argument("--sweep", action="store_true", help="--quick only: compare several preset level lists")
    ap.add_argument("--dedup", type=float, default=0.75, help="drop boxes whose IoU with a kept box exceeds this")
    ap.add_argument("--max-nodes", type=int, default=700, help="cap on nodes per graph (0 = no cap)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="parallel processes")
    ap.add_argument("--quick", action="store_true", help="test split only: report proposal count and recall, write nothing")
    ap.add_argument("--compactness", type=float, default=10.0)
    ap.add_argument("--edge-thr", type=float, default=0.30)
    ap.add_argument("--edge-topk", type=int, default=6)
    args = ap.parse_args()
    if not (STAGE / "train_index.csv").exists():
        sys.exit("Run scripts/prepare_data.py first (staging folder not found).")
    if args.quick:
        quick_report(args)
        return
    PUB.mkdir(parents=True, exist_ok=True)

    tr_nodes, tr_edges, tr_gt = build_split("train", args)
    te_nodes, te_edges, te_gt = build_split("test", args)

    te_nodes = te_nodes.drop(columns=["label", "best_iou"])
    te_nodes.insert(0, "id", np.arange(len(te_nodes)))
    tr_nodes.to_csv(PUB / "train_nodes.csv", index=False)
    tr_edges.to_csv(PUB / "train_edges.csv", index=False)
    tr_gt.to_csv(PUB / "train_boxes.csv", index=False)
    te_nodes.to_csv(PUB / "test_nodes.csv", index=False)
    te_nodes[["id"]].to_csv(PUB / "test_ids.csv", index=False)
    te_edges.to_csv(PUB / "test_edges.csv", index=False)
    pd.read_csv(PRIV / "classes.csv").to_csv(PUB / "classes.csv", index=False)
    pd.DataFrame({"id": te_nodes["id"], "class_id": 0, "score": 0.0}).to_csv(PUB / "sample_submission.csv", index=False)

    node_rows = te_nodes[["id", "image_id"] + BOX].assign(kind="node", class_id=-1)
    gt_rows = te_gt.assign(kind="gt", id=-1)
    labels = pd.concat([gt_rows, node_rows], ignore_index=True)[["kind", "id", "image_id", "class_id"] + BOX]
    labels.to_csv(PRIV / "test_labels.csv", index=False)

    # ceiling: perfect node classification (score = true best IoU of nodes that match a GT box)
    te_full = build_oracle(te_nodes, te_gt)
    oracle = calculate_metrics(detections_from_nodes(te_full, labels), te_gt.astype({"class_id": int}))
    te_boxes = {i: g[BOX].to_numpy(float) for i, g in te_nodes.groupby("image_id")}
    recall = recall_report(te_boxes, te_gt)
    stats = {
        "num_node_features": len(FEATURES),
        "train_graphs": int(tr_nodes["image_id"].nunique()), "test_graphs": int(te_nodes["image_id"].nunique()),
        "train_nodes": int(len(tr_nodes)), "test_nodes": int(len(te_nodes)),
        "train_edges": int(len(tr_edges)), "test_edges": int(len(te_edges)),
        "mean_nodes_per_graph": round(len(tr_nodes) / tr_nodes["image_id"].nunique(), 1),
        "train_foreground_node_rate": round(float((tr_nodes["label"] >= 0).mean()), 4),
        "test_gt_recall_at_iou50_by_proposals": recall,
        "test_ceiling_map50_perfect_node_scores": round(oracle["map50"], 4),
        "test_ceiling_map50_95": round(oracle["map50_95"], 4),
    }
    (PUB / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print("\nNext: python scripts/generate_keys.py (first time only), then python scripts/encrypt_labels.py")


def recall_report(boxes_by_img: dict, gt: pd.DataFrame) -> dict:
    """Share of ground-truth boxes covered at IoU>=0.5 by some proposal, overall and by box size."""
    hit, area = [], []
    for iid, g in gt.groupby("image_id"):
        gb = g[BOX].to_numpy(float)
        hit += list((box_iou(gb, boxes_by_img[iid]).max(1) >= 0.5) if len(boxes_by_img.get(iid, [])) else np.zeros(len(gb), bool))
        area += list(np.sqrt((gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])))
    hit, area = np.array(hit), np.array(area)
    bins = {"small (<48 px)": area < 48, "medium (48-128 px)": (area >= 48) & (area < 128), "large (>=128 px)": area >= 128}
    out = {"overall": round(float(hit.mean()), 4)}
    out.update({k: f"{hit[m].mean():.3f} of {int(m.sum())}" for k, m in bins.items() if m.any()})
    return out


def quick_report(args) -> None:
    import time
    index = pd.read_csv(STAGE / "test_index.csv", dtype={"image_id": str})
    gt = pd.read_csv(STAGE / "test_boxes.csv", dtype={"image_id": str})
    if args.limit:
        index = index.head(args.limit)
        gt = gt[gt["image_id"].isin(set(index["image_id"]))]
    paths = [STAGE / "test" / "images" / f for f in index["file_name"]]
    caps = [args.max_nodes] if not args.sweep else [0, args.max_nodes]
    configs = [(parse_levels(c), m) for c in SWEEP for m in caps] if args.sweep else [(args.levels, args.max_nodes)]
    for levels, cap in configs:
        args.levels, args.max_nodes = levels, cap
        t0 = time.time()
        res = list(run_jobs(paths, args, False))
        boxes = {iid: r[0][:, :4] for iid, r in zip(index["image_id"], res)}
        print(json.dumps({"levels": ",".join(f"{n}:{int(c)}" if c else str(n) for n, c in levels), "max_nodes": cap,
                          "images": len(index), "sec_per_image": round((time.time() - t0) / len(index), 2),
                          "raw_proposals_per_image": round(float(np.mean([r[3] for r in res])), 1),
                          "nodes_per_image": round(float(np.mean([len(r[0]) for r in res])), 1),
                          "recall_at_iou50": recall_report(boxes, gt)}, indent=2), flush=True)


def build_oracle(te_nodes: pd.DataFrame, te_gt: pd.DataFrame) -> pd.DataFrame:
    """Predictions of an oracle: true class with score = best IoU for nodes matching a GT box."""
    rows = []
    gt_by = {k: g for k, g in te_gt.groupby("image_id")}
    for iid, g in te_nodes.groupby("image_id"):
        lab, best = label_nodes(g[BOX].to_numpy(float), gt_by.get(iid, te_gt.iloc[:0]))
        m = lab >= 0
        rows.append(pd.DataFrame({"id": g["id"].to_numpy()[m], "class_id": lab[m], "score": best[m]}))
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    main()

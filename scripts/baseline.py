"""Baseline for SignDetectArena: a 3-layer GraphSAGE-style GNN on the proposal graphs (pure PyTorch, CPU).

    python scripts/baseline.py
    python scripts/baseline.py --epochs 10 --limit 40      # quick smoke test

What it does
  * every training image is a graph of candidate boxes; all graphs are stacked into one big graph
  * each layer mixes a node's own features with the MEAN of its neighbours' features (self and
    neighbour weights are separate, so a rare object box is not averaged away by its background neighbours)
  * two heads: a class head (background + object classes) and an IoU head (how well the box fits an object)
  * training targets use near-misses: a box is an object box if its IoU with a ground-truth box is >= 0.4
    (computed from train_boxes.csv); the IoU head regresses best_iou
  * score of class c for a node = P(class c) * predicted IoU
15% of the train graphs are held out to report a validation mAP@0.5.
Writes predictions.csv (id,class_id,score). Validate before encrypting:
    python competition/validate_submission.py predictions.csv
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "data" / "public"
sys.path.insert(0, str(ROOT / "competition"))
from metrics import box_iou, calculate_metrics, detections_from_nodes  # noqa: E402

BOX = ["x_min", "y_min", "x_max", "y_max"]
META = {"image_id", "node_id", "id", "label", "best_iou"}
LOG_COLS = ["lap_var", "grad_std", "n_parts"]
NEAR_IOU = 0.4


def load_graph(nodes_csv: str, edges_csv: str, limit: int = 0):
    nodes = pd.read_csv(PUB / nodes_csv, dtype={"image_id": str})
    edges = pd.read_csv(PUB / edges_csv, dtype={"image_id": str})
    if limit:
        keep = sorted(nodes["image_id"].unique())[:limit]
        nodes, edges = nodes[nodes["image_id"].isin(keep)], edges[edges["image_id"].isin(keep)]
    nodes = nodes.reset_index(drop=True)
    offset = nodes.reset_index().groupby("image_id")["index"].min()
    off = edges["image_id"].map(offset).to_numpy()
    src = edges["source"].to_numpy() + off
    dst = edges["target"].to_numpy() + off
    ei = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    return nodes, ei


def features(nodes: pd.DataFrame, cols):
    X = nodes[cols].copy()
    for c in LOG_COLS:
        X[c] = np.log1p(X[c])
    return X.to_numpy(np.float32)


def mean_adj(ei: np.ndarray, n: int) -> torch.Tensor:
    """Row-normalised adjacency (mean over neighbours, no self loops) as a CSR tensor."""
    deg = np.bincount(ei[0], minlength=n).astype(np.float32)
    w = 1.0 / np.maximum(deg[ei[0]], 1.0)
    return torch.sparse_coo_tensor(torch.from_numpy(ei), torch.from_numpy(w.astype(np.float32)), (n, n)).coalesce().to_sparse_csr()


def near_targets(nodes: pd.DataFrame, gt: pd.DataFrame):
    """Class of the best-overlapping ground-truth box (if IoU >= NEAR_IOU, else -1) for every train node."""
    gt_by = {k: g for k, g in gt.groupby("image_id")}
    cls = np.full(len(nodes), -1, dtype=np.int64)
    best = nodes["best_iou"].to_numpy(np.float32)
    for iid, idx in nodes.groupby("image_id").indices.items():
        g = gt_by.get(iid)
        if g is None:
            continue
        iou = box_iou(nodes[BOX].to_numpy(float)[idx], g[BOX].to_numpy(float))
        j = iou.argmax(1)
        ok = iou[np.arange(len(idx)), j] >= NEAR_IOU
        cls[idx[ok]] = g["class_id"].to_numpy()[j[ok]]
    return cls, best


class SAGE(nn.Module):
    def __init__(self, d_in, n_cls, hidden=64, layers=3, dropout=0.2):
        super().__init__()
        dims = [d_in] + [hidden] * layers
        self.self_lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.nb_lins = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(dims[:-1], dims[1:]))
        self.cls, self.iou = nn.Linear(hidden, n_cls), nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, a):
        for s, nb in zip(self.self_lins, self.nb_lins):
            x = F.dropout(x, self.dropout, self.training)
            x = F.relu(s(x) + nb(torch.sparse.mm(a, x)))
        return self.cls(x), self.iou(x).squeeze(-1)


def scores_from(logits, iou_hat) -> np.ndarray:
    return (torch.softmax(logits, 1)[:, 1:] * torch.sigmoid(iou_hat)[:, None]).numpy()


def to_predictions(ids, score: np.ndarray, top_classes=3, min_score=0.002) -> pd.DataFrame:
    """score: (n, K). Keep each node's best `top_classes` classes above min_score."""
    order = np.argsort(-score, axis=1)[:, :top_classes]
    r = np.repeat(np.arange(len(score)), order.shape[1])
    c = order.ravel()
    s = score[r, c]
    keep = s >= min_score
    return pd.DataFrame({"id": np.asarray(ids)[r[keep]], "class_id": c[keep], "score": np.round(s[keep], 5)})


def val_map(score, nodes: pd.DataFrame, gt: pd.DataFrame) -> float:
    labels = nodes[["image_id"] + BOX].assign(kind="node", id=np.arange(len(nodes)), class_id=-1)
    dets = detections_from_nodes(to_predictions(np.arange(len(nodes)), score), labels)
    return calculate_metrics(dets.astype({"class_id": int}), gt)["map50"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N train graphs")
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    tr, tr_ei = load_graph("train_nodes.csv", "train_edges.csv", args.limit)
    te, te_ei = load_graph("test_nodes.csv", "test_edges.csv")
    cols = [c for c in tr.columns if c not in META]
    K = int(pd.read_csv(PUB / "classes.csv").shape[0])
    gt_all = pd.read_csv(PUB / "train_boxes.csv", dtype={"image_id": str})

    X = features(tr, cols)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xtr = torch.from_numpy((X - mu) / sd)
    Xte = torch.from_numpy((features(te, cols) - mu) / sd)
    near_cls, best = near_targets(tr, gt_all)
    ycls = torch.from_numpy(near_cls + 1)  # 0 = background
    yiou = torch.from_numpy(best)
    A_tr, A_te = mean_adj(tr_ei, len(tr)), mean_adj(te_ei, len(te))

    ids = sorted(tr["image_id"].unique())
    val_ids = set(np.random.default_rng(0).choice(ids, max(1, int(0.15 * len(ids))), replace=False))
    is_val = torch.from_numpy(tr["image_id"].isin(val_ids).to_numpy())
    cnt = torch.bincount(ycls[~is_val], minlength=K + 1).float().clamp(min=1)
    weight = (cnt[0] / cnt).sqrt().clamp(max=30.0)  # background = 1, rarer classes up to 30
    print("near-object node counts (bg, classes):", cnt.int().tolist())
    gt_val = gt_all[gt_all["image_id"].isin(val_ids)].astype({"class_id": int})
    val_nodes = tr[is_val.numpy()].reset_index(drop=True)

    model = SAGE(len(cols), K + 1, args.hidden, args.layers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        logits, iou_hat = model(Xtr, A_tr)
        loss = F.cross_entropy(logits[~is_val], ycls[~is_val], weight=weight) \
            + 5.0 * F.binary_cross_entropy_with_logits(iou_hat[~is_val], yiou[~is_val])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (epoch + 1) % 25 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                lg, ih = model(Xtr, A_tr)
            m = val_map(scores_from(lg[is_val], ih[is_val]), val_nodes, gt_val)
            print(f"epoch {epoch + 1}/{args.epochs}  loss {loss.item():.4f}  val_mAP50 {m:.4f}  "
                  f"elapsed {(time.time() - t0) / 60:.2f} min", flush=True)

    model.eval()
    with torch.no_grad():
        lg, ih = model(Xte, A_te)
    to_predictions(te["id"].to_numpy(), scores_from(lg, ih)).to_csv(args.out, index=False)
    print(f"Wrote {args.out}. Total time {(time.time() - t0) / 60:.2f} min")


if __name__ == "__main__":
    main()

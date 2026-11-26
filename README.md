# SignDetectArena - Object Detection Challenge (Graph Neural Networks)

## 🎯 Challenge Overview
**Can your graph neural network find and classify road signs in street photos, using only a CPU?**

SignDetectArena is a **graph-learning** competition built on a modified version of the
Kaggle [Road Sign Detection](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection) dataset. Each photo is turned into a graph of **candidate
boxes**: nodes are boxes found by merging superpixels, edges connect overlapping or nested boxes, and
every node carries 71 hand-crafted features. **No raw pixels are published.** For every test node you
predict how likely its box is an object of each of 4 road-sign classes. The scorer turns your node scores
into detections (per-class non-maximum suppression) and reports **mAP@0.5**, with **mAP@[0.5:0.95]**
as the secondary metric.

The competition is part of a GNN benchmark that compares LLM coding agents with human participants.
Every submission is tagged `human`, `llm` or `hybrid`. **Your model must be a graph neural network
that uses the edges** (see Rules).

---

## 📊 Dataset Information

Derived from the Road Sign Detection dataset (877 street photos, Pascal VOC annotations) and **modified so it differs from the original**. Each image became one graph of candidate boxes with the following properties:

* **Undirected:** True
* **Self-loops:** False
* **Total Graphs:** 809 (546 train + 263 test)
* **Total Nodes:** 566,300 (382,200 train + 184,100 test), exactly 700 per graph
* **Total Node Features:** 71
* **Total Edges:** 3,035,363 (2,047,437 train + 987,926 test)

* **Task:** classify candidate boxes (background or one of 4 classes), scored as object detection
* **Classes:** speed limit, traffic light, stop, crosswalk (`data/public/classes.csv`)
* **Graphs:** one per image, a few hundred nodes each (see `data/public/dataset_stats.json`)
* **Node features (71):** box position and size, aspect ratio, region fill ratio, number of merged superpixels, mean and std of R/G/B, hue, saturation and brightness statistics, colour histograms (hue 8 bins, saturation 4, brightness 4), gradient-direction histogram (8 bins), local texture pattern histogram (10 bins), edge density, contrast against the surrounding ring (colour, edge strength, histogram distance), distance to the image's mean colour, image width and height
* **Edges:** undirected, between boxes with IoU or containment of at least 0.3 (up to 8 per node), with the pair's `iou`
* **Graph ids:** shuffled and renamed `img_00001...` (original file names are hidden)
* **Train / test:** split by image, roughly 70 / 30
* **Node labels (train):** `label` is the class of the ground-truth box with the highest IoU if that IoU is at least 0.5, else `-1` (background); `best_iou` is also given

### Modifications (applied to the images before the graphs are built)
| Modification | Where | Effect |
|:--|:--|:--|
| Class imbalance | train only | Images that contain one of the 2 rarest classes are kept with probability 0.5, so rare classes become rarer. The test set keeps the natural distribution. |
| Geometric changes | all images, stronger on test | Random crop-and-resize, horizontal / vertical flip and 90-degree rotation. Boxes are transformed with the image; a box is dropped only if less than 40% of it stays visible. |
| Colour jitter | all images, stronger on test | Random brightness, contrast, saturation and hue shifts. |
| Image noise | 50% of test, 20% of train (milder) | Gaussian noise, Gaussian blur or heavy JPEG compression. |
| Resize | all | Longest side 416 px. |

### Protection against matching the original source
Road Sign Detection is public, so a model or agent could try to recognise a test image and look up
its original labels. To make that harder, the geometric changes, colour jitter and noise above were applied to the
images, every graph was renamed in shuffled order, the train/test split is new, and only candidate-box
graphs with summary features are released (no pixels). These steps **reduce** the chance that a graph
can be matched to its public source. They do **not** make it impossible, and a model that saw this
dataset during pretraining may still have an advantage. Participants must not look up the original dataset
(see Rules).

---

## 🔥 What Makes This Hard?

- ✅ **Extreme node imbalance** - under 1% of candidate boxes are real signs; the rest are background
- ✅ **Class imbalance** - speed-limit signs dominate; stop signs and traffic lights are rare in training
- ✅ **Many overlapping boxes per object** - a good model scores the best box highly and uses neighbours to suppress the rest
- ✅ **Noisy, rotated test images** - noise, blur, compression and rotations that training rarely shows
- ✅ **Candidate-box ceiling** - about 16% of test signs (mostly the smallest) have no overlapping candidate box, so even perfect scores reach 0.8594

**Expected Performance (mAP@0.5):**
- Random scores: ~0.00
- GraphSAGE-style baseline: ~0.16
- Good GNN: >0.2 🎯
- Ceiling (perfect node scores): 0.8594

---

## 📈 Dataset Distribution

| Split | Graphs | Nodes | Edges | Sign-node rate |
|:------|:-------|:------|:------|:---------------|
| **Training** | 546 | 382,200 | 2,047,437 | 0.7% |
| **Testing** | 263 | 184,100 | 987,926 | hidden |

Numbers come from `data/public/dataset_stats.json`.

------|:------|:------|
| Train | see stats | `train_nodes.csv` (features + labels), `train_edges.csv`, `train_boxes.csv` (ground-truth boxes) |
| Test  | see stats | `test_nodes.csv` (features only), `test_edges.csv`; labels hidden (`private/test_labels.csv`, encrypted copy in `encryption/`) |

---

## 📋 Quick Links

- **Submit:** open a Pull Request (see section 4)
- **Leaderboard:** https://ignatiusbalayo.github.io/SignDetectArena/leaderboard.html
- **Download dataset:** https://github.com/ignatiusbalayo/SignDetectArena/releases/download/v1.0/SignDetectArena_Dataset.zip
- **Submissions guide:** [submissions/README.md](submissions/README.md)

---

## 1. Repository Structure
```
.
├── data/
│   ├── public/
│   │   ├── train_nodes.csv             # image_id, node_id, 71 features, label, best_iou
│   │   ├── train_edges.csv             # image_id, source, target, iou
│   │   ├── train_boxes.csv             # image_id, class_id, x_min, y_min, x_max, y_max
│   │   ├── test_nodes.csv              # id, image_id, node_id, 71 features (no labels)
│   │   ├── test_edges.csv              # image_id, source, target, iou
│   │   ├── classes.csv                 # class_id, name
│   │   ├── dataset_stats.json
│   │   └── sample_submission.csv       # id, class_id, score
│   ├── raw/                            # original Road Sign Detection (git-ignored)
│   └── private/                        # staging images + test_labels.csv (git-ignored)
├── encryption/
│   ├── public_key.pem                  # participants encrypt with this
│   └── test_labels.enc                 # encrypted private labels
├── competition/
│   ├── config.yaml
│   ├── validate_submission.py
│   ├── evaluate.py
│   ├── metrics.py                      # node-to-detection conversion, NMS, COCO-style mAP
│   ├── crypto_utils.py
│   ├── score_pr.py                     # CI orchestrator
│   └── render_leaderboard.py
├── leaderboard/                        # leaderboard.csv / leaderboard.md
├── docs/                               # GitHub Pages leaderboard
├── scripts/
│   ├── baseline.py                     # 3-layer GraphSAGE-style baseline (pure PyTorch)
│   ├── encrypt_submission.py           # participants
│   ├── prepare_data.py                 # organizers (modify images)
│   ├── build_graphs.py                 # organizers (images -> graphs)
│   ├── generate_keys.py                # organizers
│   ├── encrypt_labels.py               # organizers
│   └── make_release.py                 # organizers
├── submissions/inbox/                  # one folder per team (via Pull Request)
├── ORGANIZERS.md                       # setup checklist for the organizers
└── .github/workflows/score_submission.yml
```

---

## 2. Dataset Files

### train_nodes.csv
```csv
image_id,node_id,x_min,y_min,x_max,y_max,cx,cy,...,n_parts,label,best_iou
img_00001,0,12.0,40.0,96.0,130.0,0.10,0.17,...,3,-1,0.12
```
`label` is `-1` for background, otherwise the class id (0 to 3). `best_iou` is the highest IoU with any ground-truth box.

### train_edges.csv / test_edges.csv
```csv
image_id,source,target,iou
img_00001,0,1,0.43
```
`source` and `target` are `node_id`s inside the same graph. Each undirected pair appears once.

### test_nodes.csv
Same columns without `label` / `best_iou`, plus a global `id` that you must use in your submission.

### train_boxes.csv
The ground-truth boxes of the training images (pixels, in the same coordinates as the node boxes).

**See `data/public/sample_submission.csv` for the required output format.**

---

## 3. 📥 Download & Setup
```bash
git clone https://github.com/ignatiusbalayo/SignDetectArena.git
cd SignDetectArena

# Download dataset (GitHub Release)
wget https://github.com/ignatiusbalayo/SignDetectArena/releases/download/v1.0/SignDetectArena_Dataset.zip
unzip SignDetectArena_Dataset.zip

pip install -r requirements.txt

# Run the baseline (CPU)
python scripts/baseline.py
```

---

## 4. 📤 How to Submit

Submissions are **encrypted** so nobody can copy another team's predictions.

1. Create `predictions.csv` (format below) and validate it:
   ```bash
   python competition/validate_submission.py predictions.csv
   ```
2. Encrypt it and create your submission folder:
   ```bash
   python scripts/encrypt_submission.py predictions.csv \
       --team my_team --model "GraphSAGE" --type human --notes "3 layers, hidden 128"
   ```
   `--type` is `human`, `llm` or `hybrid`.
3. Commit **only** `submissions/inbox/my_team/`, push to your fork, and open a Pull Request to `main`.
4. The scoring bot decrypts, scores and updates the leaderboard automatically, then comments on your PR.

### Required format
```csv
id,class_id,score
17,3,0.91
17,5,0.04
52,0,0.77
```
One row per (node, class) you want to score; `id` comes from `test_nodes.csv`, `class_id` is 0 to 3 and `score` is in [0, 1].
Each pair appears at most once, and a header-only file is valid (it scores 0).

### Scoring
Every row becomes a detection with the box of its node. For each image and class, the scorer applies
non-maximum suppression (IoU 0.5) on its 300 best-scoring nodes, drops scores below 0.001 and keeps the 100 best detections per image.
mAP is then COCO-style (101-point AP, averaged over classes) at IoU 0.5, and averaged over 0.5 to 0.95 for the secondary metric.

PRs that change anything other than your own `submissions/inbox/<team>/` folder are rejected.

---

## 5. 🏆 Leaderboard

**Live:** https://ignatiusbalayo.github.io/SignDetectArena/leaderboard.html

- Ranked by mAP@0.5 (Kaggle-style: ties share a rank)
- Search by team or model, filter by human / llm / hybrid
- Updated automatically after every scored submission

---

## 6. Getting Started

### Explore the Data
```python
import pandas as pd

nodes = pd.read_csv("data/public/train_nodes.csv")
edges = pd.read_csv("data/public/train_edges.csv")
test = pd.read_csv("data/public/test_nodes.csv")

g = nodes[nodes.image_id == nodes.image_id.iloc[0]]
print(len(g), "nodes;", (g.label >= 0).sum(), "sign nodes")
print("Test nodes to score:", len(test))
```

### Build a Simple GNN
All graphs are stacked into one big disconnected graph (needs `pip install torch-geometric`).
```python
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch_geometric.nn import SAGEConv

K = 4
feat = [c for c in nodes.columns if c not in ("image_id", "node_id", "id", "label", "best_iou")]

def to_edge_index(n, e):
    row = pd.Series(range(len(n)), index=pd.MultiIndex.from_frame(n[["image_id", "node_id"]]))
    s = row.loc[pd.MultiIndex.from_frame(e[["image_id", "source"]])].to_numpy()
    t = row.loc[pd.MultiIndex.from_frame(e[["image_id", "target"]])].to_numpy()
    return torch.tensor(np.stack([np.r_[s, t], np.r_[t, s]]))      # undirected

x = torch.tensor(nodes[feat].values, dtype=torch.float)
mu, sd = x.mean(0), x.std(0) + 1e-6
x = (x - mu) / sd
edge_index = to_edge_index(nodes, edges)
y = torch.tensor(nodes["label"].values + 1)                        # 0 = background

class Net(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.c1, self.c2, self.out = SAGEConv(d, h), SAGEConv(h, h), nn.Linear(h, K + 1)
    def forward(self, x, ei):
        return self.out(self.c2(self.c1(x, ei).relu(), ei).relu())

model = Net(x.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=0.01)
w = torch.ones(K + 1); w[1:] = 20.0                               # up-weight the rare sign classes
for epoch in range(100):
    model.train()
    loss = nn.functional.cross_entropy(model(x, edge_index), y, weight=w)
    opt.zero_grad(); loss.backward(); opt.step()
```

### Generate Predictions
```python
test_edges = pd.read_csv("data/public/test_edges.csv")
xt = (torch.tensor(test[feat].values, dtype=torch.float) - mu) / sd
model.eval()
with torch.no_grad():
    prob = torch.softmax(model(xt, to_edge_index(test, test_edges)), 1)[:, 1:].numpy()

r, c = np.nonzero(prob >= 0.01)                                   # one row per (node, class)
pd.DataFrame({"id": test["id"].values[r], "class_id": c, "score": prob[r, c]}).to_csv("predictions.csv", index=False)

# Validate before submitting!
# python competition/validate_submission.py predictions.csv
```

---

## 7. Rules

### Allowed
- Any graph neural network (GCN, GraphSAGE, GAT, GIN, graph transformers, ...), any GNN library, and publicly available pretrained weights
- Feature engineering on the provided node features, graph augmentation, loss design, score calibration, post-processing of scores
- Large files (`train_nodes.csv`, `train_edges.csv`, `test_nodes.csv`, `test_edges.csv`) come from the Release zip, not from git
- Ensembles, unlimited offline training

### Not Allowed
- **Models that ignore the graph.** The model must use the edges (message passing or graph attention). A per-node MLP or gradient-boosting model on node features alone is not a valid entry
- Using the original Road Sign Detection images or annotations (including their test data) to train
- Looking up, downloading or matching against the original Road Sign Detection dataset, or any other copy of it
- Manual labeling of test nodes
- Test-set peeking or probing the scorer
- **ONE submission per team** (enforced by the bot)

Graph-use is checked from the `--notes` / model description and by reviewers; it cannot be verified from predictions alone.

---

## 8. Baseline Performance

**3-layer GraphSAGE-style GNN (hidden 64, dropout 0.2), full-batch, 150 epochs on CPU, with a class head and an IoU head; score = class probability x predicted IoU.**

| Metric | Value |
|:--|:--|
| mAP@0.5 | 0.1579 |
| mAP@[0.5:0.95] | 0.0780 |
| CPU time | about 29 minutes (organizer's laptop CPU) |

Perfect node scores reach 0.8594 mAP@0.5 and 0.5699 mAP@[0.5:0.95] (`dataset_stats.json`), so the baseline is a floor to beat, not a ceiling. Scores come from one run on one CPU and vary slightly with hardware.

---

## 9. Tips for Success
- Almost every node is background: weight the loss, or sample negatives.
- Several boxes cover the same object; neighbours on the graph tell you which one is best. Edge-aware models (GAT, GraphSAGE) help.
- Use `best_iou` as a regression target to rank boxes inside an object.
- Test images are noisier than train: normalise features per graph or add feature noise.
- Rare classes (stop, traffic light) have few examples: try focal loss or class-balanced sampling.
- Signs have characteristic colours and shapes: the colour histograms and fill ratio are strong clues.

---

## 10. Resources
- Kipf & Welling, *Semi-Supervised Classification with GCNs*, ICLR 2017
- Hamilton et al., *GraphSAGE*, NeurIPS 2017; Velickovic et al., *GAT*, ICLR 2018
- Uijlings et al., *Selective Search for Object Recognition*, IJCV 2013 (the idea behind the candidate boxes)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io), [Road Sign Detection](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection)

---

## 11. References
- **Dataset:** Road Sign Detection, Larxel (Kaggle), Pascal VOC annotations. Check the dataset's licence on its Kaggle page before redistributing anything derived from it.
- **Superpixels:** Achanta et al., *SLIC Superpixels*, IEEE TPAMI 2012.

---

## 12. License
Code: MIT License. Dataset: derived from Road Sign Detection and subject to its original licence.

---

**Good luck!** 🚀

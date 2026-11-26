"""ORGANIZER ONLY - build the modified Road Sign Detection images for SignDetectArena (step 1 of 2).

Raw input (not committed) - the Kaggle "Road Sign Detection" dataset (Pascal VOC format):
    data/raw/RoadSign/images/road*.png          street photos
    data/raw/RoadSign/annotations/road*.xml     one XML per image with <object><name> and <bndbox>

What this script does (all seeded, fully reproducible):
  1. Keep the K most frequent classes (default 4, i.e. all) and remap them to 0..K-1.
  2. Orient images by EXIF, resize so the longest side is 416 px, rescale boxes.
  3. Rename every image to img_00001... in shuffled order (hides original file names).
  4. Split by image into train / test (default 70 / 30).
  5. MODIFICATION 1 - class imbalance: images that contain a tail-class object are
     kept in TRAIN with probability --tail-keep, making rare classes rarer.
     The TEST split keeps its natural distribution.
  6. MODIFICATION 2 - anti-matching geometry (labels transformed too): random crop-and-
     resize, horizontal / vertical flip and 90-degree rotation. Stronger on test than train.
  7. MODIFICATION 3 - colour jitter: brightness, contrast, saturation and hue shifts.
  8. MODIFICATION 4 - noise: a fraction of test (50%) and train (20%) images get one of
     Gaussian noise, Gaussian blur or heavy JPEG compression.

Modifications 2-4 exist so that test images cannot easily be matched back to the public
Road Sign Detection images (and their labels) by file name, hashing or simple feature
matching. They reduce leakage; they do not make it impossible.

Outputs (all PRIVATE and git-ignored - raw pixels are never published):
    data/private/staging/{train,test}/images/*.jpg
    data/private/staging/{train,test}_index.csv    image_id,file_name,width,height
    data/private/staging/{train,test}_boxes.csv    image_id,class_id,x_min,y_min,x_max,y_max
    data/private/classes.csv
    data/private/modification_log.json             (source file + applied changes per image)
    data/private/prep_stats.json

Next step: python scripts/build_graphs.py   (turns the staged images into the public graph files)

Usage:
    python scripts/prepare_data.py --raw data/raw/RoadSign
"""

import argparse
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
BOX_COLS = ["x_min", "y_min", "x_max", "y_max"]

# Strength of the geometric / colour changes per split (test is stronger on purpose).
GEOM = {
    "train": {"crop_min": 0.85, "p_hflip": 0.5, "p_vflip": 0.10, "p_rot90": 0.15, "color": 0.15},
    "test": {"crop_min": 0.70, "p_hflip": 0.5, "p_vflip": 0.25, "p_rot90": 0.50, "color": 0.30},
}
MIN_VISIBLE = 0.40   # keep a cropped box only if at least 40% of it stays visible
MIN_SIDE = 4         # drop boxes thinner than this many pixels


def corrupt(img: Image.Image, rng: np.random.Generator, mild: bool) -> tuple[Image.Image, str]:
    kind = str(rng.choice(["gauss", "blur", "jpeg"]))
    if kind == "gauss":
        sigma = rng.uniform(8, 16) if mild else rng.uniform(15, 30)
        arr = np.asarray(img, dtype=np.float32) + rng.normal(0, sigma, (img.height, img.width, 3))
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    elif kind == "blur":
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.8, 1.2) if mild else rng.uniform(1.5, 2.5)))
    else:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=int(rng.integers(30, 45) if mild else rng.integers(10, 25)))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img, kind


def color_jitter(img: Image.Image, rng: np.random.Generator, strength: float) -> Image.Image:
    """Brightness / contrast / saturation factors in [1-s, 1+s] and a hue shift of up to +-s*40 units."""
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        img = enhancer(img).enhance(rng.uniform(1 - strength, 1 + strength))
    h, s, v = img.convert("HSV").split()
    shift = int(rng.uniform(-strength, strength) * 40)
    h = Image.fromarray(((np.asarray(h).astype(np.int16) + shift) % 256).astype(np.uint8))
    return Image.merge("HSV", (h, s, v)).convert("RGB")


def geometric(img: Image.Image, boxes: list, rng: np.random.Generator, cfg: dict):
    """Random crop-and-resize, flips and 90-degree rotation, applied to image AND boxes.

    boxes: list of (class_id, x1, y1, x2, y2) in pixels. Returns (img, boxes, description).
    Boxes that are less than MIN_VISIBLE visible after cropping are dropped.
    """
    w, h = img.size
    desc = []
    new_boxes = boxes
    for _ in range(10):  # retry until at least one box survives the crop
        k = np.sqrt(rng.uniform(cfg["crop_min"] ** 2, 1.0))  # side scale
        cw, ch = w * k, h * k
        x0, y0 = rng.uniform(0, w - cw), rng.uniform(0, h - ch)
        kept = []
        for c, x1, y1, x2, y2 in boxes:
            nx1, ny1, nx2, ny2 = (x1 - x0) * w / cw, (y1 - y0) * h / ch, (x2 - x0) * w / cw, (y2 - y0) * h / ch
            full = max(1e-6, (nx2 - nx1) * (ny2 - ny1))
            cx1, cy1, cx2, cy2 = max(0.0, nx1), max(0.0, ny1), min(float(w), nx2), min(float(h), ny2)
            if cx2 - cx1 < MIN_SIDE or cy2 - cy1 < MIN_SIDE:
                continue
            if (cx2 - cx1) * (cy2 - cy1) / full >= MIN_VISIBLE:
                kept.append((c, cx1, cy1, cx2, cy2))
        if kept:
            img = img.crop((round(x0), round(y0), round(x0 + cw), round(y0 + ch))).resize((w, h), Image.BILINEAR)
            new_boxes = kept
            desc.append(f"crop{k:.2f}")
            break
    boxes = new_boxes
    if rng.random() < cfg["p_hflip"]:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        boxes = [(c, w - x2, y1, w - x1, y2) for c, x1, y1, x2, y2 in boxes]
        desc.append("hflip")
    if rng.random() < cfg["p_vflip"]:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        boxes = [(c, x1, h - y2, x2, h - y1) for c, x1, y1, x2, y2 in boxes]
        desc.append("vflip")
    if rng.random() < cfg["p_rot90"]:
        img = img.transpose(Image.ROTATE_90)  # counter-clockwise; image becomes h x w
        boxes = [(c, y1, w - x2, y2, w - x1) for c, x1, y1, x2, y2 in boxes]
        desc.append("rot90")
    return img, boxes, "+".join(desc) or "none"


def load_voc(raw: Path) -> dict:
    """Read the Pascal VOC XML files into a COCO-like dict (file_name is relative to `raw`)."""
    import xml.etree.ElementTree as ET
    xmls = sorted((raw / "annotations").glob("*.xml"))
    if not xmls:
        raise SystemExit(f"No XML files in {raw / 'annotations'}")
    cats, images, anns = {}, [], []
    for i, xp in enumerate(xmls):
        root = ET.parse(xp).getroot()
        fname = root.findtext("filename") or (xp.stem + ".png")
        images.append({"id": i, "file_name": f"images/{fname}"})
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            bb = obj.find("bndbox")
            if not name or bb is None:
                continue
            x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            cid = cats.setdefault(name, len(cats))
            anns.append({"image_id": i, "category_id": cid, "bbox": [x1, y1, x2 - x1, y2 - y1]})
    return {"categories": [{"id": v, "name": k} for k, v in cats.items()], "images": images, "annotations": anns}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data/raw/RoadSign"))
    ap.add_argument("--classes", type=int, default=4, help="keep the top-K categories")
    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--tail-keep", type=float, default=0.5)
    ap.add_argument("--tail-classes", type=int, default=2, help="number of rarest kept classes")
    ap.add_argument("--noise-test", type=float, default=0.50)
    ap.add_argument("--noise-train", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    raw = Path(args.raw)
    coco = load_voc(raw)
    names = {c["id"]: c["name"] for c in coco["categories"]}
    counts = Counter(a["category_id"] for a in coco["annotations"])
    top = [cid for cid, _ in counts.most_common(args.classes)]
    remap = {cid: i for i, cid in enumerate(top)}  # class 0 = most frequent
    tail = set(range(args.classes - args.tail_classes, args.classes))

    anns_by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        if a["category_id"] in remap:
            anns_by_img.setdefault(a["image_id"], []).append(a)
    images = [im for im in coco["images"] if im["id"] in anns_by_img]
    order = rng.permutation(len(images))
    images = [images[i] for i in order]
    print(f"{len(images)} images with at least one of the top-{args.classes} classes")

    n_test = int(round(len(images) * args.test_frac))
    split_of = {im["id"]: ("test" if i < n_test else "train") for i, im in enumerate(images)}

    priv = ROOT / "data" / "private"
    out = priv / "staging"
    for s in ("train", "test"):
        (out / s / "images").mkdir(parents=True, exist_ok=True)
    priv.mkdir(parents=True, exist_ok=True)

    train_rows, label_rows, img_rows = [], [], []
    mod_log, noise_counts = {}, Counter()
    next_id = 1
    for im in images:
        split = split_of[im["id"]]
        src = raw / im["file_name"]
        if not src.exists():
            print(f"  missing image, skipped: {src}")
            continue
        pil = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        ow, oh = pil.width, pil.height  # annotation space = EXIF-oriented image
        scale = args.size / max(ow, oh)
        nw, nh = max(1, round(ow * scale)), max(1, round(oh * scale))

        boxes = []
        for a in anns_by_img[im["id"]]:
            x, y, w, h = a["bbox"]
            x1, y1 = max(0.0, x * scale), max(0.0, y * scale)
            x2, y2 = min(float(nw), (x + w) * scale), min(float(nh), (y + h) * scale)
            if x2 - x1 >= MIN_SIDE and y2 - y1 >= MIN_SIDE:
                boxes.append((remap[a["category_id"]], x1, y1, x2, y2))
        if not boxes:
            continue

        # MODIFICATION 1: make tail classes rarer in TRAIN (whole images, so labels stay consistent)
        if split == "train" and any(b[0] in tail for b in boxes) and rng.random() > args.tail_keep:
            continue

        pil = pil.resize((nw, nh), Image.BILINEAR)
        # MODIFICATIONS 2 + 3: geometry (labels follow) and colour jitter
        pil, boxes, geom_desc = geometric(pil, boxes, rng, GEOM[split])
        pil = color_jitter(pil, rng, GEOM[split]["color"])
        # MODIFICATION 4: noise / corruption
        frac = args.noise_test if split == "test" else args.noise_train
        if rng.random() < frac:
            pil, kind = corrupt(pil, rng, mild=(split == "train"))
        else:
            kind = "none"

        image_id = f"img_{next_id:05d}"
        next_id += 1
        fw, fh = pil.size
        pil.save(out / split / "images" / f"{image_id}.jpg", quality=95)
        mod_log[image_id] = {"split": split, "source": im["file_name"], "geometry": geom_desc, "noise": kind}
        noise_counts[(split, kind)] += 1
        img_rows.append({"split": split, "image_id": image_id, "file_name": f"{image_id}.jpg", "width": fw, "height": fh})
        for c, x1, y1, x2, y2 in boxes:
            row = {"image_id": image_id, "class_id": c, "x_min": round(x1, 1), "y_min": round(y1, 1),
                   "x_max": round(x2, 1), "y_max": round(y2, 1)}
            (train_rows if split == "train" else label_rows).append(row)

    img_df = pd.DataFrame(img_rows)
    cols = ["image_id", "class_id"] + BOX_COLS
    icols = ["image_id", "file_name", "width", "height"]
    pd.DataFrame(train_rows, columns=cols).to_csv(out / "train_boxes.csv", index=False)
    pd.DataFrame(label_rows, columns=cols).to_csv(out / "test_boxes.csv", index=False)
    img_df[img_df["split"] == "train"][icols].to_csv(out / "train_index.csv", index=False)
    img_df[img_df["split"] == "test"][icols].to_csv(out / "test_index.csv", index=False)
    pd.DataFrame({"class_id": range(len(top)), "name": [names[c] for c in top]}).to_csv(
        priv / "classes.csv", index=False)

    tr, te = pd.DataFrame(train_rows), pd.DataFrame(label_rows)
    geom_counts = Counter(
        part for v in mod_log.values() for part in v["geometry"].split("+") if part != "none" and not part.startswith("crop"))
    stats = {
        "seed": args.seed,
        "train_images": int((img_df["split"] == "train").sum()),
        "test_images": int((img_df["split"] == "test").sum()),
        "train_boxes_per_class": {names[top[c]]: int(n) for c, n in tr["class_id"].value_counts().sort_index().items()},
        "test_boxes_per_class": {names[top[c]]: int(n) for c, n in te["class_id"].value_counts().sort_index().items()},
        "noisy_images": {f"{s}:{k}": v for (s, k), v in sorted(noise_counts.items())},
        "geometry_flip_rotate_counts": dict(geom_counts),
    }
    (priv / "prep_stats.json").write_text(json.dumps(stats, indent=2))
    # Private audit trail: original file name and every change applied to each image.
    (priv / "modification_log.json").write_text(json.dumps(mod_log, indent=2))
    print(json.dumps(stats, indent=2))
    print("\nNext: python scripts/build_graphs.py")


if __name__ == "__main__":
    main()

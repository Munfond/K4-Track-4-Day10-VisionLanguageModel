"""Label conversion and dataset validation (usable without a GPU)."""
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image
from scripts.labcommon import CLASSES, read_jsonl


def yolo_to_box(label, yolo, width=640, height=360, coord="norm1000"):
    if coord not in ("norm1000", "pixel"):
        raise ValueError(coord)
    if label not in CLASSES:
        raise ValueError(f"Unknown class: {label}")
    xc, yc, w, h = yolo
    sx, sy = (1000, 1000) if coord == "norm1000" else (width, height)
    values = [(xc-w/2)*sx, (yc-h/2)*sy, (xc+w/2)*sx, (yc+h/2)*sy]
    box = [max(0, min(scale, round(v))) for v, scale in zip(values, (sx, sy, sx, sy))]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"Degenerate rounded box: {label} {yolo} -> {box}")
    return {"label": label, "bbox_2d": box}


def objects_to_answer(objects, width=640, height=360, coord="norm1000"):
    boxes = [yolo_to_box(o["label"], o["yolo"], width, height, coord) for o in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0], b["label"]))
    return json.dumps(boxes, separators=(",", ":"))


def make_prompt(coord="norm1000", width=640, height=360):
    scale = ("integer coordinates normalized to 0-1000 relative to image width and height"
             if coord == "norm1000" else f"integer pixel coordinates for this {width}x{height} image")
    return ("Locate every pothole, crack, and manhole in the road image. "
            "Return ONLY a JSON list. Each object must have label (pothole, crack, or manhole) "
            'and bbox_2d [x1,y1,x2,y2] with ' + scale + ". "
            "Use top-left and bottom-right corners. Sort boxes top to bottom, then left to right. "
            "Return [] if no object is present. Do not include markdown or explanation.")


def load_rows(data_dir, split, fraction=1.0, seed=42, limit=None):
    rows = read_jsonl(Path(data_dir) / "splits" / f"{split}.jsonl")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if fraction < 1:
        # Same seeded ordering for 10% and 100%; subset is reproducible.
        rows = random.Random(seed).sample(rows, max(1, round(len(rows)*fraction)))
    return rows[:limit] if limit is not None else rows


def open_image(row, data_dir):
    with Image.open(Path(data_dir) / row["image_path"]) as im:
        return im.convert("RGB")


def validate_splits(data_dir):
    stats, sessions, image_ids = {}, {}, {}
    meta = json.loads((Path(data_dir)/"splits/meta.json").read_text(encoding="utf-8"))
    for split in ("train", "val", "test"):
        rows = load_rows(data_dir, split)
        assert rows, f"Empty {split}"
        ids = [r["image_id"] for r in rows]
        assert len(ids) == len(set(ids)), f"Duplicate IDs in {split}"
        sessions[split] = {r["session"] for r in rows}
        image_ids[split] = set(ids)
        counts = Counter(o["label"] for r in rows for o in r["objects"])
        stats[split] = {"images": len(rows), "boxes": {c: counts[c] for c in CLASSES}}
        assert stats[split] == meta["counts"][split], f"Metadata mismatch: {split}"
        for row in rows:
            assert (Path(data_dir)/row["image_path"]).is_file(), row["image_id"]
            objects_to_answer(row["objects"], row["width"], row["height"])
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not sessions[a] & sessions[b], f"Session leakage {a}/{b}"
        assert not image_ids[a] & image_ids[b], f"Image leakage {a}/{b}"
    return {"dataset_version": meta["dataset_version"], "counts": stats}

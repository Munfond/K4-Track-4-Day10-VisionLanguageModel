"""Generate real-data illustrations and a report table, never invented scores."""
import json
from pathlib import Path

from PIL import Image, ImageDraw
from scripts.labcommon import read_jsonl
from scripts.score import _gt_boxes, _match, _to_norm1000, parse_output
from .data import load_rows, objects_to_answer, open_image


def draw_boxes(draw, boxes, width, height, color, prefix, y_offset=40):
    for item in boxes:
        x1, y1, x2, y2 = item["bbox"]
        bb = (x1*width/1000, y1*height/1000+y_offset, x2*width/1000, y2*height/1000+y_offset)
        draw.rectangle(bb, outline=color, width=2)
        draw.text((bb[0], max(y_offset, bb[1]-12)), f"{prefix}:{item['label']}", fill=color)


def panel(row, data_dir, pred=None, caption="Ground truth"):
    image = open_image(row, data_dir)
    canvas = Image.new("RGB", (image.width, image.height+40), "#101820")
    canvas.paste(image, (0, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), caption, fill="white")
    draw.text((8, 22), "GREEN = ground truth | RED = prediction", fill="white")
    draw_boxes(draw, _gt_boxes(row), image.width, image.height, "#00ff66", "GT")
    if pred is not None:
        draw_boxes(draw, pred, image.width, image.height, "#ff4040", "P")
    return canvas


def draw_ground_truth(rows, data_dir, out):
    out.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        # Quantization error must be <= 0.5 normalized unit before scaling to pixels.
        rounded = parse_output(objects_to_answer(row["objects"], row["width"], row["height"]))
        original = sorted(_gt_boxes(row), key=lambda b: (b["bbox"][1], b["bbox"][0], b["label"]))
        # Compare as a multiset; rounding can tie adjacent rows and reorder them.
        for gt in original:
            j = next((j for j, r in enumerate(rounded) if r["label"] == gt["label"]
                      and all(abs(a-b) <= .500001 for a, b in zip(r["bbox"], gt["bbox"]))), None)
            assert j is not None, "YOLO round-trip mismatch"
            rounded.pop(j)
        panel(row, data_dir, parse_output(objects_to_answer(row["objects"], row["width"], row["height"])),
              f"YOLO -> norm1000 -> pixel | sample {i+1}").save(out/f"sample_{i+1}.png")


def visualize(pred_file, data_dir, out):
    pred_file, out = Path(pred_file), Path(out)
    manifest = json.loads(pred_file.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    rows = load_rows(data_dir, manifest["split"])
    predictions = {r["image_id"]: r["raw"] for r in read_jsonl(pred_file)}
    groups = {"correct": [], "error": []}
    for row in rows:
        if row["image_id"] not in predictions:
            continue
        raw = predictions[row["image_id"]]
        parsed = parse_output(raw)
        pred = _to_norm1000(parsed or [], manifest["coord"], row["width"], row["height"])
        per, _ = _match(_gt_boxes(row), pred)
        tp, fp, fn = (sum(v[k] for v in per.values()) for k in range(3))
        kind = "correct" if parsed is not None and fp == fn == 0 else "error"
        groups[kind].append((row, pred, tp, fp, fn, parsed is not None, raw))
    selected = groups["correct"][:3] + groups["error"][:3]
    used = {item[0]["image_id"] for item in selected}
    for item in groups["correct"] + groups["error"]:
        if len(selected) >= 6:
            break
        if item[0]["image_id"] not in used:
            selected.append(item)
            used.add(item[0]["image_id"])
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Prediction examples", "", "Green: ground truth. Red: predictions.", "",
             "Examples are selected for illustration, not an estimate of accuracy.", ""]
    for i, (row, pred, tp, fp, fn, valid, raw) in enumerate(selected, 1):
        filename = f"example_{i}.png"
        caption = f"TP={tp} FP={fp} FN={fn} valid_json={valid}"
        panel(row, data_dir, pred, caption).save(out/filename)
        lines += [f"## {i}. {row['image_id']}", "", caption, "", f"![{caption}]({filename})", "",
                  "Raw output:", "", "```text", raw, "```", ""]
    (out/"index.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {len(selected)} examples in {out}; correct={len(groups['correct'])}, error={len(groups['error'])}")


def summarize(evidence_dir, out):
    files = sorted(Path(evidence_dir).glob("*.json"))
    lines = ["# Results from score.py", "", "Copy this table into REPORT.md and write your own analysis.", "",
             "E0 minutes/VRAM refer to inference; trained setups refer to training. E0 training time is zero.", "",
             "| Run | Trainable params | Peak VRAM GB | Minutes | F1 | CI95 | FP/image | pothole F1 | crack F1 | manhole F1 |",
             "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for file in files:
        r = json.loads(file.read_text(encoding="utf-8"))
        m = r["meta"]
        ci = r["f1_ci95"]
        cells = [r["name"], str(m.get("trainable_params", 0)), str(m.get("peak_vram_gb", "-")),
                 str(m.get("inference_minutes", m.get("train_minutes", "-"))), f"{r['micro']['f1']:.4f}", f"[{ci[0]:.4f}, {ci[1]:.4f}]",
                 f"{r['fp_per_image']:.4f}"] + [f"{r['per_class'][c]['f1']:.4f}" for c in ("pothole", "crack", "manhole")]
        lines.append("| " + " | ".join(cells) + " |")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))

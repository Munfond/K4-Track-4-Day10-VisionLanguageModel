"""Read final artifacts and independently verify the report against actual evidence.

Run from the repo root: python -m vlm_lab.submission_audit
Instructor scripts/tests are never modified.
"""
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

from PIL import Image
import yaml

from scripts.labcommon import read_jsonl
from scripts.score import _gt_boxes, _match, _prf, parse_output

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("e0", "e1", "e2", "e3", "e4_data_size_0.1", "e4_data_size_1.0")


def number(text):
    return float(text.strip().removesuffix("*").replace(",", "."))


def audit():
    results = []

    def check(name, ok, detail=""):
        results.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    ev = {k: json.loads((ROOT/f"report/evidence/{k}.json").read_text(encoding="utf-8")) for k in KEYS}
    split_meta = json.loads((ROOT/"report/split_meta.json").read_text(encoding="utf-8"))
    gt_counts = split_meta["counts"]["test"]["boxes"]
    for key, value in ev.items():
        check(f"{key} schema", value["score_version"] == 1 and value["split"] == "test"
              and value["coord"] == "norm1000" and value["n_images"] == 311
              and value["missing_predictions"] == 0 and value["dataset_version"] == split_meta["dataset_version"])
        micro = value["micro"]
        sums = [sum(p[c] for p in value["per_class"].values()) for c in ("tp", "fp", "fn")]
        check(f"{key} count arithmetic", sums == [micro[c] for c in ("tp", "fp", "fn")]
              and all(p["tp"]+p["fn"] == gt_counts[c] for c, p in value["per_class"].items() if c != "other"))
        f1 = 2*micro["tp"] / (2*micro["tp"]+micro["fp"]+micro["fn"])
        check(f"{key} metrics arithmetic", math.isclose(f1, micro["f1"])
              and math.isclose(micro["fp"]/311, value["fp_per_image"])
              and value["f1_ci95"][0] <= micro["f1"] <= value["f1_ci95"][1])
        class_ok = True
        for stats in value["per_class"].values():
            recomputed = _prf(stats["tp"], stats["fp"], stats["fn"])
            class_ok &= all(math.isclose(stats[m], recomputed[m]) for m in ("precision", "recall", "f1"))
        check(f"{key} per-class arithmetic", class_ok)
    check("E1 rubric thresholds", ev["e1"]["parse_valid_rate"] >= .95
          and ev["e1"]["micro"]["f1"]-ev["e0"]["micro"]["f1"] >= .10)
    a, b = dict(ev["e3"]), dict(ev["e4_data_size_1.0"])
    a.pop("name"); b.pop("name")
    check("E4 100% reuses E3 exactly", a == b)
    facts = json.loads((ROOT/"report/diagnostics/notebook_run_facts.json").read_text(encoding="utf-8"))
    for run in facts["runs"]:
        key = run["config"]["name"].lower()
        cfg = yaml.safe_load((ROOT/f"configs/{key}.yaml").read_text(encoding="utf-8"))
        check(f"{key} actual config", all(run["config"][k] == v for k, v in cfg.items())
              and run["step"] == 300 and run["complete"])
        check(f"{key} resource logs match evidence", all(run[k] == ev[key]["meta"][k]
              for k in ("trainable_params", "peak_vram_gb", "train_minutes")))
    check("E4 base chosen on val", facts["e4"]["base"] == max(facts["e4"]["val_f1"], key=facts["e4"]["val_f1"].get))
    report = (ROOT/"REPORT.md").read_text(encoding="utf-8")
    check("Report identity and 10 answers", "Nguyễn Đức Anh" in report and "2A202602625" in report
          and "<điền" not in report and len(re.findall(r"^### Câu \d+", report, re.M)) == 10)
    rows = [s for s in report.splitlines() if s.startswith("| E")]
    table_ok = len(rows) == 6
    for row, key in zip(rows, KEYS):
        cells = [s.strip() for s in row.strip("|").split("|")]
        value = ev[key]
        expected = [value["meta"]["peak_vram_gb"], value["meta"].get("inference_minutes", value["meta"]["train_minutes"]),
                    value["micro"]["f1"], value["fp_per_image"]] + [value["per_class"][c]["f1"] for c in ("pothole", "crack", "manhole")]
        actual = [number(cells[i]) for i in (3, 4, 5, 7, 8, 9, 10)]
        table_ok &= int(cells[2].replace(".", "")) == value["meta"]["trainable_params"]
        table_ok &= all(abs(x-y) <= .000051 for x, y in zip(actual, expected))
        ci = [number(s) for s in cells[6].strip("[]").split(";")]
        table_ok &= all(abs(x-y) <= .000051 for x, y in zip(ci, value["f1_ci95"]))
    check("All report table numbers match JSON", table_ok)
    files = list((ROOT/"report/figures").glob("e*/example_*.png"))
    image_ok = True
    for path in files:
        with Image.open(path) as image:
            image.verify()
    check("At least 6 readable illustrations", image_ok and len(files) >= 6, f"{len(files)} PNG files")
    entries_path = ROOT/"data/splits/test.jsonl"
    if entries_path.exists():
        entries = {e["image_id"]: e for e in read_jsonl(entries_path)}
        counts_ok, n = True, 0
        for index in (ROOT/"report/figures").glob("e*/index.md"):
            text = index.read_text(encoding="utf-8")
            for image_id, tp, fp, fn, raw in re.findall(
                r"## \d+\. ([^\n]+)\n\s*TP=(\d+) FP=(\d+) FN=(\d+).*?```text\n(.*?)\n```", text, re.S):
                per, _ = _match(_gt_boxes(entries[image_id.strip()]), parse_output(raw) or [])
                actual = [sum(p[i] for p in per.values()) for i in range(3)]
                counts_ok &= actual == list(map(int, (tp, fp, fn)))
                n += 1
        check("Illustration captions match original GT/scorer", counts_ok and n == 18, f"{n} captions verified")
    original = subprocess.run(["git", "diff", "--exit-code", "c2f3997", "--", "scripts", "tests", "configs", "requirements-colab.txt"], cwd=ROOT, capture_output=True)
    check("Instructor files unchanged", original.returncode == 0)
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    bad = [s for s in tracked if s.startswith(("data/", "work/")) or s.split("/")[-1] == ".env"
           or s.endswith((".safetensors", ".bin", ".pt", ".ckpt"))]
    check("No forbidden tracked files", not bad, ", ".join(bad))
    nb = json.loads((ROOT/"lab_day10.ipynb").read_text(encoding="utf-8"))
    notebook_text = "\n".join("".join(c["source"]) + "\n" + "\n".join("".join(o.get("text", [])) for o in c.get("outputs", [])) for c in nb["cells"])
    literal_secret = re.search(r"(?:KAGGLE_KEY|HF_TOKEN)['\"]?\]?\s*[:=]\s*['\"][A-Za-z0-9_-]{20,}", notebook_text)
    check("No literal API key in notebook source/output", literal_secret is None)
    failed = [r["check"] for r in results if not r["pass"]]
    artifact = {"review_date": "2026-10-14", "dataset_version": split_meta["dataset_version"],
                "notebook_sha256_lf": hashlib.sha256((ROOT/"lab_day10.ipynb").read_text(encoding="utf-8").encode("utf-8")).hexdigest(),
                "checks": results, "failures": failed}
    (ROOT/"report/submission_audit.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"\n{len(failed)} audit FAIL")
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(audit())

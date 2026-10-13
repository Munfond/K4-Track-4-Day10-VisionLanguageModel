import json

from PIL import Image
from scripts.labcommon import write_jsonl
from scripts.score import evaluate
from vlm_lab.data import objects_to_answer
from vlm_lab.reporting import draw_ground_truth, summarize, visualize


def test_real_artifact_pipeline_on_synthetic_fixture(tmp_path):
    data = tmp_path/"data"
    (data/"images").mkdir(parents=True)
    rows, predictions = [], []
    for i in range(6):
        Image.new("RGB", (640, 360)).save(data/f"images/{i}.jpg")
        row = {"image_id": str(i), "image_path": f"images/{i}.jpg", "width": 640, "height": 360,
               "objects": [{"label": "crack", "yolo": [.5, .5, .2, .2]}]}
        rows.append(row)
        predictions.append({"image_id": str(i), "raw": objects_to_answer(row["objects"]) if i < 3 else "[]"})
    write_jsonl(data/"splits/test.jsonl", rows)
    pred_file = tmp_path/"pred.jsonl"
    write_jsonl(pred_file, predictions)
    pred_file.with_suffix(".manifest.json").write_text(json.dumps({"split": "test", "coord": "norm1000"}))
    visualize(pred_file, data, tmp_path/"figures")
    assert len(list((tmp_path/"figures").glob("*.png"))) == 6
    assert "TP=1 FP=0 FN=0" in (tmp_path/"figures/index.md").read_text()
    draw_ground_truth(rows, data, tmp_path/"ground_truth")
    score = evaluate(rows, {r["image_id"]: r["raw"] for r in predictions}, n_boot=20)
    score.update(name="fixture-only", meta={})
    evidence = tmp_path/"evidence"
    evidence.mkdir()
    (evidence/"sample.json").write_text(json.dumps(score))
    summarize(evidence, tmp_path/"results.md")
    assert "0.6667" in (tmp_path/"results.md").read_text()

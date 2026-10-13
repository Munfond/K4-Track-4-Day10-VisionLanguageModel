import json

import pytest
from vlm_lab.data import make_prompt, objects_to_answer, yolo_to_box
from scripts.score import evaluate


def test_coordinate_conversion_and_scoring_round_trip():
    objects = [{"label": "pothole", "yolo": [.5, .5, .2, .2]}]
    row = {"image_id": "one", "width": 640, "height": 360, "objects": objects}
    norm = objects_to_answer(objects)
    pixel = objects_to_answer(objects, coord="pixel")
    assert json.loads(norm)[0]["bbox_2d"] == [400, 400, 600, 600]
    assert json.loads(pixel)[0]["bbox_2d"] == [256, 144, 384, 216]
    assert evaluate([row], {"one": norm}, n_boot=20)["micro"]["f1"] == 1
    assert evaluate([row], {"one": pixel}, coord="pixel", n_boot=20)["micro"]["f1"] == 1
    assert evaluate([row], {"one": norm}, coord="pixel", n_boot=20)["micro"]["f1"] == 0


def test_empty_clipping_and_order():
    assert objects_to_answer([]) == "[]"
    assert yolo_to_box("crack", [0, 0, .2, .2])["bbox_2d"] == [0, 0, 100, 100]
    objects = [{"label": "crack", "yolo": [.7, .8, .1, .1]},
               {"label": "manhole", "yolo": [.4, .2, .1, .1]}]
    assert json.loads(objects_to_answer(objects))[0]["label"] == "manhole"
    assert yolo_to_box("crack", [.5, .5, .00001, .00001])["bbox_2d"] == [500, 500, 501, 501]
    assert "0-1000" in make_prompt()
    assert "640x360" in make_prompt("pixel")


def test_zero_height_annotation_from_colab_does_not_break_answers():
    invalid = {"label": "crack", "yolo": [.639062, .786111, .003125, 0.0]}
    valid = {"label": "pothole", "yolo": [.5, .5, .2, .2]}
    original = json.dumps([invalid, valid])
    for coord in ("norm1000", "pixel"):
        assert yolo_to_box(invalid["label"], invalid["yolo"], coord=coord) is None
        assert objects_to_answer([invalid], coord=coord) == "[]"
        assert len(json.loads(objects_to_answer([invalid, valid], coord=coord))) == 1
    assert json.dumps([invalid, valid]) == original, "Source annotations must not be mutated"
    assert yolo_to_box("crack", [.5, .5, 0, .2]) is None
    assert yolo_to_box("crack", [2, 2, .1, .1]) is None
    row = {"image_id": "bad-label", "width": 640, "height": 360, "objects": [invalid, valid]}
    official = evaluate([row], {"bad-label": objects_to_answer(row["objects"])}, n_boot=20)
    assert official["micro"]["tp"] == 1 and official["micro"]["fn"] == 1


def test_split_audit_reports_bad_labels_without_changing_original_counts(tmp_path):
    from PIL import Image
    from scripts.labcommon import write_jsonl
    from vlm_lab.data import validate_splits
    (tmp_path/"images").mkdir()
    counts = {}
    for split in ("train", "val", "test"):
        Image.new("RGB", (640, 360)).save(tmp_path/f"images/{split}.jpg")
        row = {"image_id": split, "session": split, "image_path": f"images/{split}.jpg",
               "width": 640, "height": 360,
               "objects": [{"label": "crack", "yolo": [.639062, .786111, .003125, 0.0]}]}
        write_jsonl(tmp_path/f"splits/{split}.jsonl", [row])
        counts[split] = {"images": 1, "boxes": {"pothole": 0, "crack": 1, "manhole": 0}}
    (tmp_path/"splits/meta.json").write_text(json.dumps({"dataset_version": "fixture", "counts": counts}))
    summary = validate_splits(tmp_path)
    assert summary["counts"] == counts
    for split in counts:
        assert summary["annotation_quality"][split]["skipped_target_boxes"] == 1
        assert summary["annotation_quality"][split]["skipped_by_class"] == {"crack": 1}
        assert json.loads((tmp_path/f"splits/{split}.jsonl").read_text())["objects"][0]["yolo"][3] == 0


def test_positive_tiny_boxes_and_edges_stay_parseable():
    from scripts.score import parse_output
    for center in (0, .5, 1):
        obj = {"label": "crack", "yolo": [center, center, .00001, .00001]}
        for coord in ("norm1000", "pixel"):
            parsed = parse_output(objects_to_answer([obj], coord=coord))
            assert parsed is not None and len(parsed) == 1

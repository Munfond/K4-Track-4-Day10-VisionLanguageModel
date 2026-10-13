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
    with pytest.raises(ValueError, match="Degenerate"):
        yolo_to_box("crack", [.5, .5, .00001, .00001])
    assert "0-1000" in make_prompt()
    assert "640x360" in make_prompt("pixel")

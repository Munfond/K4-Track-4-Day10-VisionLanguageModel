import json

import pytest
from vlm_lab.reward import iou_reward

GT = [{"label": "pothole", "bbox": [100, 100, 200, 200]}]
PRED = {"label": "pothole", "bbox_2d": [100, 100, 200, 200]}


def test_perfect_invalid_wrong_class_extra_and_empty():
    assert iou_reward(json.dumps([PRED]), GT) == 1
    assert iou_reward("not JSON", GT) == 0
    assert iou_reward(json.dumps([{**PRED, "label": "crack"}]), GT) == .2
    assert iou_reward(json.dumps([PRED, PRED]), GT) == pytest.approx(.6)
    assert iou_reward("[]", []) == 1
    assert iou_reward("[]", GT) == .2
    assert iou_reward(json.dumps([PRED]), []) == .2


def test_huge_box_and_duplicate_gt_cannot_be_rewarded_twice():
    huge = {**PRED, "bbox_2d": [0, 0, 1000, 1000]}
    assert iou_reward(json.dumps([huge]), GT) == .2
    assert iou_reward(json.dumps([PRED]), GT*2) == pytest.approx(.6)

"""Optional E5 reward building block; does not claim that GRPO has been run."""
from scripts.score import iou_xyxy, parse_output


def iou_reward(completion, gt_boxes):
    """gt_boxes: scorer-style dictionaries with label and bbox, in the same scale."""
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0
    pairs = sorted(((iou_xyxy(g["bbox"], p["bbox"]), i, j)
                    for i, g in enumerate(gt_boxes) for j, p in enumerate(pred)
                    if g["label"] == p["label"]), reverse=True)
    used_g, used_p, total = set(), set(), 0.0
    for iou, i, j in pairs:
        if iou < .5:
            break
        if i not in used_g and j not in used_p:
            total += iou
            used_g.add(i)
            used_p.add(j)
    return .2 + .8*total/max(len(gt_boxes), len(pred), 1)

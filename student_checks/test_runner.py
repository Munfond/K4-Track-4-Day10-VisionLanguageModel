import argparse
import copy
import json

from vlm_lab.runner import batches_for_steps, config_from_args, resolve_checkpoint


def args(config, **kwargs):
    return argparse.Namespace(config=str(config), coord=None, train_fraction=None, rank=None,
                              max_steps=None, batch_size=None, grad_accum=None, **kwargs)


def test_e4_saved_coordinate_and_fraction_are_not_reset(tmp_path):
    path = tmp_path/"e4.yaml"
    path.write_text("name: E4\ncoord: pixel\ntrain_fraction: 0.1\nlora: {r: 4, alpha: 8}\n")
    cfg = config_from_args(args(path))
    assert cfg["coord"] == "pixel" and cfg["train_fraction"] == .1
    assert cfg["lora"]["r"] == 4


def test_batch_resume_matches_uninterrupted_stream():
    rows = list(range(7))
    stream = batches_for_steps(rows, 2, 42)
    full = [next(stream) for _ in range(15)]
    resumed = batches_for_steps(rows, 2, 42, start_micro=5)
    assert [next(resumed) for _ in range(10)] == full[5:]
    assert sorted(sum(full[:4], [])) == rows


def test_resolve_latest_checkpoint(tmp_path):
    checkpoint = tmp_path/"step-0100"
    checkpoint.mkdir()
    (checkpoint/"run.json").write_text("{}")
    (tmp_path/"latest.json").write_text(json.dumps({"checkpoint": checkpoint.name}))
    assert resolve_checkpoint(tmp_path) == checkpoint

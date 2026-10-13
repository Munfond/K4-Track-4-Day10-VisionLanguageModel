"""CLI: python -m vlm_lab.runner --help. Each GPU command runs in its own process."""
import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import yaml

from scripts.labcommon import read_jsonl
from .data import load_rows, validate_splits
from .modeling import apply_setup, collate_examples, encode_example, is_projector, load_base, load_projector


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dataset_version(data_dir):
    return read_json(Path(data_dir)/"splits/meta.json")["dataset_version"]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def config_from_args(args):
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg.setdefault("seed", 42)
    cfg["coord"] = args.coord or cfg.get("coord", "norm1000")
    cfg["train_fraction"] = args.train_fraction if args.train_fraction is not None else cfg.get("train_fraction", 1.)
    for key in ("max_steps", "batch_size", "grad_accum"):
        value = getattr(args, key, None)
        if value is not None:
            if value < 1:
                raise ValueError(f"{key} must be positive")
            cfg[key] = value
    if args.rank is not None:
        if args.rank < 1:
            raise ValueError("rank must be positive")
        cfg["lora"].update(r=args.rank, alpha=2*args.rank)
    return cfg


def seed_all(seed):
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batches_for_steps(rows, batch_size, seed, start_micro=0):
    """Deterministic epoch permutations, including exact restart at a microbatch."""
    per_epoch = math.ceil(len(rows)/batch_size)
    micro = start_micro
    while True:
        epoch, offset = divmod(micro, per_epoch)
        indices = list(range(len(rows)))
        random.Random(seed + epoch).shuffle(indices)
        for batch_no in range(offset, per_epoch):
            selected = indices[batch_no*batch_size:(batch_no+1)*batch_size]
            yield [rows[i] for i in selected]
            micro += 1


def save_checkpoint(model, processor, optimizer, scaler, cfg, stats, root, step, minutes, peak, version):
    import torch
    root = Path(root)
    previous_checkpoint = (root/read_json(root/"latest.json")["checkpoint"]
                           if (root/"latest.json").exists() else None)
    target = root/f"step-{step:04d}"
    staging = root/f"step-{step:04d}.incomplete"
    staging.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(staging/"adapter")
    processor.save_pretrained(staging/"processor")
    if cfg["train_projector"]:
        projector = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if is_projector(n)}
        torch.save(projector, staging/"projector.pt")
        del projector
    torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()},
               staging/"training_state.pt")
    run = {"config": cfg, "dataset_version": version, "step": step,
           "complete": step == cfg["max_steps"], "trainable_params": stats["trainable_params"],
           "projector_params": stats["projector_params"], "peak_vram_gb": peak,
           "train_minutes": minutes}
    dump_json(staging/"run.json", run)
    if target.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}")
    staging.rename(target)
    dump_json(root/"latest.json", {"checkpoint": target.name})
    dump_json(root/"run.json", run)
    # Retain historical adapters/projectors, but optimizer/RNG only for latest resume.
    # Remove a single known file only after the new checkpoint and pointer are committed.
    if previous_checkpoint is not None:
        previous_state = previous_checkpoint/"training_state.pt"
        if previous_state.exists():
            previous_state.unlink()
    print(f"Saved {target}", flush=True)
    return target


def resolve_checkpoint(path):
    path = Path(path)
    if (path/"latest.json").exists():
        path = path/read_json(path/"latest.json")["checkpoint"]
    if not (path/"run.json").exists():
        raise FileNotFoundError(f"No complete checkpoint directory: {path}")
    return path


def train(args):
    import torch
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights
    cfg = config_from_args(args)
    if "lora" not in cfg:
        raise ValueError("Train with configs/e1.yaml, e2.yaml, or e3.yaml")
    version = dataset_version(args.data_dir)
    root = Path(args.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    saved = None
    if args.resume:
        saved = resolve_checkpoint(root)
        previous = read_json(saved/"run.json")
        assert previous["config"] == cfg, "Resume config changed; use identical arguments"
        assert previous["dataset_version"] == version, "Dataset changed"
        if previous["complete"]:
            print("Training already complete; proceed to inference.")
            return
    elif (root/"latest.json").exists() or (root/"config.json").exists():
        raise FileExistsError("Run directory exists. Use --resume or a new --run-dir.")
    dump_json(root/"config.json", cfg)
    seed_all(cfg["seed"])
    model, processor = load_base(cfg["model_id"])
    model, stats = apply_setup(model, cfg)
    dump_json(root/"modules.json", stats)
    rows = load_rows(args.data_dir, "train", cfg["train_fraction"], cfg["seed"])
    dump_json(root/"train_ids.json", [r["image_id"] for r in rows])
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(cfg["lr"]), weight_decay=0)
    scaler = torch.amp.GradScaler("cuda")
    start_step, prior_minutes, prior_peak = 0, 0., 0.
    if saved:
        result = set_peft_model_state_dict(model, load_peft_weights(saved/"adapter", device="cpu"))
        assert not result.unexpected_keys, result.unexpected_keys
        if cfg["train_projector"]:
            load_projector(model, saved/"projector.pt")
        state = torch.load(saved/"training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state
        start_step, prior_minutes, prior_peak = previous["step"], previous["train_minutes"], previous["peak_vram_gb"]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    warmup = max(1, math.ceil(.05*cfg["max_steps"]))
    for step in range(start_step+1, cfg["max_steps"]+1):
        lr_scale = step/warmup if step <= warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1, cfg["max_steps"]-warmup)))
        for group in optimizer.param_groups:
            group["lr"] = float(cfg["lr"])*lr_scale
        # GradScaler often needs to lower its initial scale on fp16. Retry the same
        # microbatches/dropout without counting skipped updates as optimizer steps.
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        for attempt in range(10):
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
            stream = batches_for_steps(rows, cfg["batch_size"], cfg["seed"], (step-1)*cfg["grad_accum"])
            accumulated_loss = 0.
            for _ in range(cfg["grad_accum"]):
                encoded = [encode_example(processor, row, args.data_dir, cfg["coord"]) for row in next(stream)]
                batch = {k: v.to("cuda") for k, v in collate_examples(encoded, processor.tokenizer.pad_token_id).items()}
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(**batch).loss/cfg["grad_accum"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}; stop and debug on val")
                accumulated_loss += loss.detach().item()
                scaler.scale(loss).backward()
                del batch, encoded, loss
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
            if not torch.isfinite(norm):
                scaler.update(new_scale=scaler.get_scale()/2)
                optimizer.zero_grad(set_to_none=True)
                print(f"Gradient overflow step={step}; retry={attempt+1}, scale={scaler.get_scale()}", flush=True)
                continue
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            break
        else:
            raise FloatingPointError(f"Repeated gradient overflow at step {step}; inspect loss/masks/lr")
        elapsed = prior_minutes + (time.perf_counter()-started)/60
        peak = max(prior_peak, torch.cuda.max_memory_allocated()/1024**3)
        with (root/"train_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "loss": accumulated_loss, "lr": optimizer.param_groups[0]["lr"],
                                "grad_norm": float(norm), "peak_vram_gb": peak, "train_minutes": elapsed})+"\n")
        if step == 1 or step % 10 == 0:
            print(f"step={step}/{cfg['max_steps']} loss={accumulated_loss:.4f} peak={peak:.2f}GB time={elapsed:.1f}min", flush=True)
        if step % args.save_every == 0 or step == cfg["max_steps"]:
            torch.cuda.synchronize()
            elapsed = prior_minutes + (time.perf_counter()-started)/60
            save_checkpoint(model, processor, optimizer, scaler, cfg, stats, root, step, elapsed, peak, version)


def inspect(args):
    import torch
    from .data import open_image, objects_to_answer
    from .reporting import draw_ground_truth
    summary = validate_splits(args.data_dir)
    dump_json("report/data_checks.json", summary)
    print(json.dumps(summary, indent=2))
    rows = load_rows(args.data_dir, "train")
    draw_ground_truth(rows[:6], args.data_dir, Path("report/figures/data"))
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_id)
    row = rows[0]
    encoded = encode_example(processor, row, args.data_dir)
    image_token = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    count = int((encoded["input_ids"] == image_token).sum())
    decoded = processor.tokenizer.decode(encoded["labels"][encoded["labels"] != -100], skip_special_tokens=False)
    print("Vision tokens:", count, "\nSupervised answer:", decoded)
    assert count == 220, f"Expected 220 image tokens, got {count}; inspect processor resolution"
    answer = objects_to_answer(row["objects"], row["width"], row["height"])
    assert decoded.startswith(answer) and "<|im_end|>" in decoded
    batch = collate_examples([encoded, encode_example(processor, rows[1], args.data_dir)], processor.tokenizer.pad_token_id)
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)
    print("Prefix, answer-only mask, image tokens and collator: PASS")


def infer(args):
    import torch
    from peft import PeftModel
    from .data import open_image
    from .modeling import conversation
    saved = resolve_checkpoint(args.checkpoint) if args.checkpoint else None
    run = read_json(saved/"run.json") if saved else None
    if run:
        assert run["dataset_version"] == dataset_version(args.data_dir), "Dataset mismatch"
        if args.split == "test" and not run["complete"]:
            raise ValueError("Incomplete training: validate on val only")
        model_id, coord = run["config"]["model_id"], run["config"]["coord"]
    else:
        model_id, coord = args.model_id, args.coord
    manifest = {"model_id": model_id, "coord": coord, "dataset_version": dataset_version(args.data_dir),
                "split": args.split, "limit": args.limit, "max_new_tokens": args.max_new_tokens,
                "checkpoint": str(saved.resolve()) if saved else None,
                "run_fingerprint": fingerprint(run) if run else None}
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sidecar = dest.with_suffix(".manifest.json")
    rows = load_rows(args.data_dir, args.split, limit=args.limit)
    done = {}
    if dest.exists():
        assert sidecar.exists() and read_json(sidecar) == manifest, "Predictions from another run; use a new output path"
        predictions = read_jsonl(dest)
        done = {r["image_id"]: r["raw"] for r in predictions}
        assert len(done) == len(predictions), "Duplicate predictions"
        assert set(done) <= {r["image_id"] for r in rows}, "Unexpected image IDs"
    dump_json(sidecar, manifest)
    remaining = [r for r in rows if r["image_id"] not in done]
    if not remaining:
        print("All predictions already saved; skipping generation.")
        return
    model, processor = load_base(model_id)
    if saved:
        model = PeftModel.from_pretrained(model, saved/"adapter", is_trainable=False)
        if run["config"]["train_projector"]:
            load_projector(model, saved/"projector.pt")
    model.eval()
    model.config.use_cache = True
    processor.tokenizer.padding_side = "left"
    resource_file = dest.with_suffix(".resources.json")
    previous_resources = read_json(resource_file) if resource_file.exists() else {}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with dest.open("a", encoding="utf-8") as f, torch.inference_mode():
        for start in range(0, len(remaining), args.batch_size):
            batch = remaining[start:start+args.batch_size]
            chats = [conversation(open_image(r, args.data_dir), r, coord) for r in batch]
            inputs = processor.apply_chat_template(chats, tokenize=True, add_generation_prompt=True,
                                                   return_dict=True, return_tensors="pt", padding=True)
            inputs.pop("token_type_ids", None)
            inputs = inputs.to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            texts = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
            for row, raw in zip(batch, texts):
                f.write(json.dumps({"image_id": row["image_id"], "raw": raw}, ensure_ascii=False)+"\n")
            f.flush()
            torch.cuda.synchronize()
            dump_json(resource_file, {
                "peak_vram_gb": max(previous_resources.get("peak_vram_gb", 0), torch.cuda.max_memory_allocated()/1024**3),
                "inference_minutes": previous_resources.get("inference_minutes", 0)+(time.perf_counter()-started)/60})
            print(f"Predictions {len(done)+min(start+args.batch_size,len(remaining))}/{len(rows)}", flush=True)


def score(args):
    manifest = read_json(Path(args.pred).with_suffix(".manifest.json"))
    if manifest["limit"] is not None:
        raise ValueError("Partial predictions are smoke tests; do not submit them as evidence")
    assert manifest["dataset_version"] == dataset_version(args.data_dir)
    entries = load_rows(args.data_dir, manifest["split"])
    predictions = read_jsonl(args.pred)
    ids = [r["image_id"] for r in predictions]
    assert len(ids) == len(set(ids)), "Duplicate predictions"
    assert set(ids) == {r["image_id"] for r in entries}, "Missing or unexpected predictions; finish inference first"
    cmd = [sys.executable, "scripts/score.py", "--pred", args.pred, "--name", args.name,
           "--out", args.out, "--split", manifest["split"], "--coord", manifest["coord"],
           "--splits-dir", str(Path(args.data_dir)/"splits")]
    if manifest["checkpoint"]:
        run = read_json(Path(manifest["checkpoint"])/"run.json")
        assert fingerprint(run) == manifest["run_fingerprint"], "Checkpoint metadata changed"
        for key in ("trainable_params", "peak_vram_gb", "train_minutes"):
            cmd += ["--meta", f"{key}={run[key]}"]
    else:
        resource_file = Path(args.pred).with_suffix(".resources.json")
        if resource_file.exists():
            resources = read_json(resource_file)
            cmd += ["--meta", "trainable_params=0", "--meta", "train_minutes=0"]
            for key, value in resources.items():
                cmd += ["--meta", f"{key}={value}"]
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("train", "infer", "inspect", "score", "summarize", "visualize", "e4-config"):
        a = sub.add_parser(name)
        a.add_argument("--data-dir", default="data")
        if name in ("train", "e4-config"):
            a.add_argument("--config", required=True)
            a.add_argument("--coord", choices=["norm1000", "pixel"])
            a.add_argument("--train-fraction", type=float)
            a.add_argument("--rank", type=int)
            for key in ("max-steps", "batch-size", "grad-accum"):
                a.add_argument("--"+key, type=int)
            if name == "train":
                a.add_argument("--run-dir", required=True)
                a.add_argument("--resume", action="store_true")
                a.add_argument("--save-every", type=int, default=100)
            else:
                a.add_argument("--out", required=True)
        if name in ("inspect", "infer"):
            a.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
        if name == "infer":
            a.add_argument("--checkpoint")
            a.add_argument("--coord", choices=["norm1000", "pixel"], default="norm1000")
            a.add_argument("--split", choices=["val", "test"], default="val")
            a.add_argument("--limit", type=int)
            a.add_argument("--out", required=True)
            a.add_argument("--batch-size", type=int, default=4)
            a.add_argument("--max-new-tokens", type=int, default=256)
        if name in ("score", "visualize"):
            a.add_argument("--pred", required=True)
            a.add_argument("--out", required=True)
        if name == "score":
            a.add_argument("--name", required=True)
        if name == "summarize":
            a.add_argument("--evidence-dir", default="report/evidence")
            a.add_argument("--out", default="report/results.md")
    args = p.parse_args()
    if args.command == "train":
        if args.save_every < 1:
            p.error("--save-every must be positive")
        train(args)
    elif args.command == "infer":
        if args.batch_size < 1 or args.limit is not None and args.limit < 1:
            p.error("batch-size and limit must be positive")
        infer(args)
    elif args.command == "inspect":
        inspect(args)
    elif args.command == "score":
        score(args)
    elif args.command == "e4-config":
        cfg = config_from_args(args)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    else:
        from .reporting import summarize, visualize
        if args.command == "summarize":
            summarize(args.evidence_dir, args.out)
        else:
            visualize(args.pred, args.data_dir, args.out)


if __name__ == "__main__":
    main()

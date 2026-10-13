"""Qwen3-VL loading, assistant-only labels, and exact LoRA targets."""
import re

from .data import make_prompt, objects_to_answer, open_image

LLM_PATTERN = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
VIT_PATTERN = r"model\.visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))"
# Transformers 5.19 matches exclusions against full paths from the root.
# "visual" does not match the descendants of "model.visual".
QUANTIZATION_SKIP_MODULES = ("model.visual", "lm_head")


def is_projector(name):
    return "visual.merger." in name or "visual.deepstack_merger_list." in name


def load_base(model_id):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime (T4). This command requires CUDA.")
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, llm_int8_skip_modules=list(QUANTIZATION_SKIP_MODULES))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quant, dtype=torch.float16,
        device_map={"": 0}, attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    for name, param in model.named_parameters():
        if "visual" in name or "lm_head" in name:
            assert param.is_floating_point() and param.__class__.__name__ not in ("Params4bit", "Int8Params"), f"Unexpected quantized parameter: {name}"
    quantized_llm = [n for n, m in model.named_modules()
                     if n.startswith("model.language_model.") and m.__class__.__name__ == "Linear4bit"]
    assert len(quantized_llm) == 252, f"Expected 252 quantized LLM Linear modules; found {len(quantized_llm)}"
    print(f"Quantization check: {len(quantized_llm)} LLM Linear4bit modules; ViT/projectors/lm_head floating point", flush=True)
    return model, processor


def conversation(image, row, coord, answer=None):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": make_prompt(coord, row["width"], row["height"])}]}]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def encode_example(processor, row, data_dir, coord="norm1000"):
    import torch
    image = open_image(row, data_dir)
    answer = objects_to_answer(row["objects"], row["width"], row["height"], coord)
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    prompt = processor.apply_chat_template(conversation(image, row, coord),
                                           add_generation_prompt=True, **kw)
    full = processor.apply_chat_template(conversation(image, row, coord, answer),
                                         add_generation_prompt=False, **kw)
    n = prompt["input_ids"].shape[1]
    assert torch.equal(full["input_ids"][:, :n], prompt["input_ids"]), "Chat prefix mismatch"
    full.pop("token_type_ids", None)
    full["labels"] = full["input_ids"].clone()
    full["labels"][:, :n] = -100
    assert (full["labels"] != -100).any(), "No supervised answer tokens"
    return dict(full)


def collate_examples(examples, pad_token_id):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    keys = set(examples[0])
    assert all(set(e) == keys for e in examples), "Inconsistent processor fields"
    result = {}
    for key in keys:
        values = [e[key] for e in examples]
        if key in ("input_ids", "attention_mask", "mm_token_type_ids", "labels"):
            pad = -100 if key == "labels" else pad_token_id if key == "input_ids" else 0
            result[key] = pad_sequence([v.squeeze(0) for v in values], batch_first=True,
                                       padding_value=pad)
        elif key in ("pixel_values", "image_grid_thw"):
            result[key] = torch.cat(values, dim=0)
        else:
            raise ValueError(f"Unhandled processor field: {key}; inspect processor output")
    return result


def apply_setup(model, config):
    import torch
    from peft import LoraConfig, get_peft_model
    use_vit = config["lora"]["targets"] == "llm+vit"
    patterns = [LLM_PATTERN] + ([VIT_PATTERN] if use_vit else [])
    matched = [n for n, _ in model.named_modules() if any(re.fullmatch(p, n) for p in patterns)]
    expected = 348 if use_vit else 252
    assert len(matched) == expected, f"Matched {len(matched)} modules; expected {expected}. Inspect model tree."
    for p in model.parameters():
        p.requires_grad = False
    # Avoid prepare_model_for_kbit_training's blanket fp32 cast of the frozen ViT/lm_head.
    # Cast language norms only, keep the large frozen components at fp16 on T4.
    for name, module in model.named_modules():
        if "language_model" in name and "norm" in name:
            module.to(torch.float32)
    model.enable_input_require_grads()
    lc = config["lora"]
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=lc["r"], lora_alpha=lc["alpha"],
        lora_dropout=lc["dropout"], target_modules=matched, bias="none"))
    if config["train_projector"]:
        for name, p in model.named_parameters():
            if is_projector(name):
                p.data = p.data.float()
                p.requires_grad = True
    # GradScaler cannot unscale fp16 gradients: every trainable parameter must be fp32.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    stats = {"matched_modules": matched,
             "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
             "projector_params": sum(p.numel() for n, p in model.named_parameters() if is_projector(n) and p.requires_grad)}
    if config["train_projector"]:
        assert stats["projector_params"] > 100_000_000, "Projector not fully unfrozen"
    print({k: v for k, v in stats.items() if k != "matched_modules"}, flush=True)
    return model, stats


def load_projector(model, checkpoint):
    import torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    params = dict(model.named_parameters())
    expected = {n for n in params if is_projector(n)}
    assert set(state) == expected, "Saved projector does not match this model"
    with torch.no_grad():
        for name, value in state.items():
            params[name].data = params[name].data.float()
            params[name].copy_(value.to(params[name].device))

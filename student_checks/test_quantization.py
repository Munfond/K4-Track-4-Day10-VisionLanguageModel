"""Regression tests against the installed Transformers/bitsandbytes APIs."""
import pytest

torch = pytest.importorskip("torch")
utils = pytest.importorskip("transformers.quantizers.quantizers_utils")
from vlm_lab.modeling import QUANTIZATION_SKIP_MODULES


def test_real_transformers_matcher_skips_visual_descendants_only():
    for name in ("model.visual", "model.visual.blocks.0.attn.qkv",
                 "model.visual.blocks.23.mlp.linear_fc2", "model.visual.merger.linear_fc1",
                 "model.visual.deepstack_merger_list.0.linear_fc2", "lm_head"):
        assert not utils.should_convert_module(name, list(QUANTIZATION_SKIP_MODULES)), name
    for name in ("model.language_model.layers.0.self_attn.q_proj",
                 "model.language_model.layers.35.mlp.down_proj"):
        assert utils.should_convert_module(name, list(QUANTIZATION_SKIP_MODULES)), name


def test_real_bnb_conversion_keeps_vit_mergers_and_head_unquantized():
    bnb = pytest.importorskip("bitsandbytes")
    pytest.importorskip("accelerate")
    from transformers import BitsAndBytesConfig
    from transformers.integrations.bitsandbytes import replace_with_bnb_linear

    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.visual = torch.nn.Module()
    root.model.visual.blocks = torch.nn.ModuleList([torch.nn.Module()])
    root.model.visual.blocks[0].attn = torch.nn.Module()
    root.model.visual.blocks[0].attn.qkv = torch.nn.Linear(8, 24).half()
    root.model.visual.merger = torch.nn.Linear(8, 8).half()
    root.model.visual.deepstack_merger_list = torch.nn.ModuleList([torch.nn.Linear(8, 8).half()])
    root.model.language_model = torch.nn.Module()
    root.model.language_model.q_proj = torch.nn.Linear(8, 8).half()
    root.lm_head = torch.nn.Linear(8, 16).half()
    cfg = BitsAndBytesConfig(load_in_4bit=True, llm_int8_skip_modules=list(QUANTIZATION_SKIP_MODULES))
    replace_with_bnb_linear(root, modules_to_not_convert=cfg.llm_int8_skip_modules, quantization_config=cfg)
    assert isinstance(root.model.language_model.q_proj, bnb.nn.Linear4bit)
    for module in (root.model.visual.blocks[0].attn.qkv, root.model.visual.merger,
                   root.model.visual.deepstack_merger_list[0], root.lm_head):
        assert type(module) is torch.nn.Linear
        assert module.weight.dtype == torch.float16

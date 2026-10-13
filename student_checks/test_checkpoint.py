"""Actual PEFT save/restore regression, on a small CPU model without Hub access."""
import json

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
from vlm_lab.runner import load_local_adapter_weights, validate_checkpoint_files


def test_saved_peft_adapter_loads_locally_and_restores_exactly(tmp_path, monkeypatch):
    from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
    from peft.utils.save_and_load import get_peft_model_state_dict
    import huggingface_hub

    def no_hub(*args, **kwargs):
        raise AssertionError("A local checkpoint must not access the Hub")

    monkeypatch.setattr(huggingface_hub, "file_exists", no_hub)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_hub)
    base = lambda: torch.nn.Sequential(torch.nn.Linear(3, 2))
    cfg = LoraConfig(r=1, lora_alpha=2, target_modules=["0"])
    original = get_peft_model(base(), cfg)
    for name, p in original.named_parameters():
        if "lora_" in name:
            p.data.fill_(.75)
    original.save_pretrained(str(tmp_path/"adapter"))
    (tmp_path/"run.json").write_text(json.dumps({"config": {"train_projector": False}}))
    expected = get_peft_model_state_dict(original)
    restored = get_peft_model(base(), cfg)
    result = set_peft_model_state_dict(restored, load_local_adapter_weights(tmp_path))
    assert not result.unexpected_keys
    actual = get_peft_model_state_dict(restored)
    assert set(actual) == set(expected)
    assert all(torch.equal(actual[k], expected[k]) for k in expected)
    path, _ = validate_checkpoint_files(tmp_path)
    assert isinstance(path, str)
    infer_model = PeftModel.from_pretrained(base(), path, is_trainable=False, local_files_only=True)
    assert all(torch.equal(get_peft_model_state_dict(infer_model)[k], expected[k]) for k in expected)


def test_missing_weights_produce_local_diagnostic_not_hub_error(tmp_path):
    (tmp_path/"adapter").mkdir()
    (tmp_path/"adapter/adapter_config.json").write_text("{}")
    (tmp_path/"run.json").write_text(json.dumps({"config": {"train_projector": False}}))
    with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
        load_local_adapter_weights(tmp_path)


def test_resume_requires_optimizer_and_projector(tmp_path):
    from safetensors.torch import save_file
    (tmp_path/"adapter").mkdir()
    (tmp_path/"adapter/adapter_config.json").write_text("{}")
    save_file({"tensor": torch.zeros(1)}, str(tmp_path/"adapter/adapter_model.safetensors"))
    (tmp_path/"run.json").write_text(json.dumps({"config": {"train_projector": True}}))
    with pytest.raises(FileNotFoundError) as error:
        validate_checkpoint_files(tmp_path, require_training_state=True)
    assert "projector.pt" in str(error.value) and "training_state.pt" in str(error.value)

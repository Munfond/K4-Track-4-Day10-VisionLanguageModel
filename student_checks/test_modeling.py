"""CPU tensor checks; run on Colab with its existing torch."""
import pytest

torch = pytest.importorskip("torch")
from vlm_lab.modeling import collate_examples, encode_example, load_projector


def encoded(ids, labels, grid):
    return {"input_ids": torch.tensor([ids]), "labels": torch.tensor([labels]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            "mm_token_type_ids": torch.zeros(1, len(ids), dtype=torch.long),
            "pixel_values": torch.ones(grid, 3), "image_grid_thw": torch.tensor([[1, grid, 1]])}


def test_collator_pads_text_not_pixels_and_masks_padding():
    batch = collate_examples([encoded([1, 2, 3], [-100, 2, 3], 4),
                              encoded([4, 5], [-100, 5], 2)], 0)
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert batch["labels"].tolist() == [[-100, 2, 3], [-100, 5, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["pixel_values"].shape == (6, 3)
    assert batch["image_grid_thw"].shape == (2, 3)


def test_mask_covers_entire_prompt_and_checks_prefix(tmp_path):
    from PIL import Image
    Image.new("RGB", (640, 360)).save(tmp_path/"one.jpg")
    row = {"image_path": "one.jpg", "width": 640, "height": 360, "objects": []}

    class Processor:
        bad = False

        def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
            ids = [1, 2, 3] if add_generation_prompt else [9 if self.bad else 1, 2, 3, 4, 5]
            return {"input_ids": torch.tensor([ids])}

    processor = Processor()
    result = encode_example(processor, row, tmp_path)
    assert result["labels"].tolist() == [[-100, -100, -100, 4, 5]]
    processor.bad = True
    with pytest.raises(AssertionError, match="prefix"):
        encode_example(processor, row, tmp_path)


def test_projector_checkpoint_restores_exact_weights_and_fp32(tmp_path):
    model = torch.nn.Module()
    model.visual = torch.nn.Module()
    model.visual.merger = torch.nn.Linear(2, 2).half()
    state = {n: torch.full_like(p, 3., dtype=torch.float32) for n, p in model.named_parameters()}
    path = tmp_path/"projector.pt"
    torch.save(state, path)
    load_projector(model, path)
    assert all(p.dtype == torch.float32 and torch.all(p == 3) for p in model.parameters())
    torch.save({}, path)
    with pytest.raises(AssertionError, match="match"):
        load_projector(model, path)

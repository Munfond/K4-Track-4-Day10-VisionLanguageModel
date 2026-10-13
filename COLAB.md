# Chạy Lab Day 10 trên Google Colab

Mã bắt buộc E0–E4 nằm trong `vlm_lab/`. Notebook `lab_day10.ipynb` gọi mã bằng subprocess,
giải phóng GPU sau mỗi lệnh. Công cụ gốc trong `scripts/` và `tests/` được giữ nguyên.
Bạn vẫn phải chạy trên GPU, kiểm tra kết quả và tự hoàn thành REPORT.md bằng số liệu thật.

`run_cmd` đọc stdout/stderr của tiến trình con và in trực tiếp trong cell Colab. Bạn sẽ thấy
`Running: ...`, log thực thi và traceback nếu lệnh lỗi. Nếu đang dùng bản notebook cũ chỉ thấy
cell kết thúc mà không có log, mở lại notebook mới từ GitHub hoặc thay hàm `run_cmd` bằng bản mới.

## 1. Đưa bản code mới lên Colab

Chọn một trong hai cách:

- **GitHub:** commit/push các file mới và notebook lên repo của bạn. Mở Colab → File → Open notebook → GitHub → dán URL repo → chọn `lab_day10.ipynb`. Sửa `REPO_URL` trong cell đầu thành URL repo chứa code mới.
- **ZIP:** mở notebook bằng tab Upload. Nén source repo (loại `.git/`, `data/`, `work/`, secrets và trọng số), upload ZIP qua Files của Colab, giải nén thành `/content/road-vlm` rồi đặt `REPO_URL = ""`. File notebook upload riêng không tự mang theo thư mục `vlm_lab/`.

Bản source đã đóng gói sẵn ở `work/colab_source.zip`. Upload file này vào Files bên trái của Colab,
rồi chạy cell sau **trước cell 0**:

```python
import zipfile
with zipfile.ZipFile('/content/colab_source.zip') as z:
    z.extractall('/content/road-vlm')
```

ZIP chứa source, notebook và công cụ gốc; không chứa dataset, secrets, trọng số hoặc số liệu thực nghiệm giả.

Mở notebook từ GitHub cũng **không tự clone repo**; cell 0 thực hiện bước đó.
Code mới hiện có trên máy local; cần push hoặc dùng ZIP trước khi cell 0 có thể tìm thấy nó.
Không dùng token GitHub trong URL hoặc cell; repo nộp được yêu cầu công khai.

## 2. Chọn GPU và chuẩn bị dữ liệu

1. Runtime → Change runtime type → Hardware accelerator → **T4 GPU**.
2. Chạy mục **0**, mount Drive và cài dependencies. Không cài lại torch của Colab nếu bản torch/torchvision có sẵn hoạt động.
3. Chạy hai bộ kiểm tra: công cụ đề và `student_checks/`. Kiểm tra tensor dùng torch có sẵn trên Colab.
4. Mở biểu tượng khóa **Secrets** bên trái, thêm `KAGGLE_USERNAME` và `KAGGLE_KEY`, bật **Notebook access**.
5. Chạy mục **1** với `DATA_MODE="kaggle"`. Nếu dùng ZIP dataset, đổi `DATA_MODE="zip"` và `DATA_PATH` thành đường dẫn ZIP; `source` dành cho thư mục đã giải nén.
6. Chạy mục **2**, đọc thống kê split và xem 6 ảnh. Cell phải xác nhận 220 token ảnh, prefix chat template, mask loss và collator.

Dataset được tải/chia lại trên ổ `/content/road-vlm/data` mỗi session mới. Version phải khớp checkpoint cũ.
Split chuẩn là train 1.422, val 276, test 311 ảnh; không tự chia lại hoặc sửa công cụ đề.

## 3. Thứ tự chạy thí nghiệm

| Mục notebook | Việc thực hiện | Kết quả |
|---|---|---|
| 3 | E0 thử 8 ảnh val, rồi đủ val/test | `work/e0*.jsonl`, `report/evidence/e0.json` |
| 4–5 | Smoke test E1 hai bước, save/load, xem target modules | Run riêng `smoke_e1`; không nộp như kết quả E1 |
| 6 | Train E1, E2, E3 độc lập từ model gốc, chấm val | Checkpoint, `run.json`, `train_log.jsonl`, điểm val |
| 7 | Suy luận/chấm test E1–E3, tạo ảnh đúng/sai | `e1.json`–`e3.json`, `report/figures/`, `report/results.md` |
| 8 | E4 mặc định 10%/100% dữ liệu | Hai evidence `e4_data_size_*.json` |
| 9 | Chẩn đoán sai thang tọa độ, đóng gói kết quả | F1 đúng/sai thang cho Câu 3, ZIP kết quả trên Drive |

Chạy từng cell và xem kết quả trước khi chuyển bước. Không nên bỏ qua smoke test.
Mặc định E1–E3 là 300 bước, batch 2, tích lũy 4. Ước lượng của đề là 20–30 phút/setup;
tốc độ thực tế phụ thuộc Colab, tải model, số box và I/O. E4 tốn thêm một hoặc hai lần train.

Trong cell 6 có thể đặt `RUNS=["e1"]` để chạy riêng E1, sau đó đổi thành E2/E3.
Cell 7 dùng cùng RUNS; trước E4 cần đã có đủ điểm val và test E1–E3.
Kiểm tra `parse_valid_rate` và loss trên **val** trước khi chấm test; không dùng test để chọn prompt/hyperparameter.

## 4. E4 chỉ đổi một biến

Cell 8 chọn setup nền bằng F1 trên **val**, rồi sao chép cấu hình thực tế của run đó.
Giữ nguyên seed, số bước, learning rate, batch, accumulation và các khối được học.

- `E4_VARIABLE="data_size"`: train thêm 10%, dùng lại kết quả 100% của setup nền.
- `E4_VARIABLE="rank"`: chạy rank 4 và 64, alpha bằng 2 lần rank. Rank 64, đặc biệt trên E3, có thể vượt VRAM T4.
- `E4_VARIABLE="coord_scale"`: dùng lại run norm1000 và train thêm pixel. Mã đổi đồng bộ nhãn, prompt và scorer.

Chọn biến trước khi xem điểm test E4. Hai giá trị có evidence riêng, cùng dataset_version.
So sánh CI95 và tài nguyên trong REPORT.md; không khẳng định một setup chắc chắn tốt hơn chỉ vì F1 cao hơn chút ít.

## 5. Dữ liệu được lưu ở đâu và tiếp tục thế nào?

Mặc định trên Google Drive: `MyDrive/road-vlm-day10/`.

```text
checkpoints/e1/ (tương tự e2, e3 và e4)
  config.json, modules.json, train_ids.json, train_log.jsonl
  latest.json, run.json
  step-0300/
    adapter/                 # LoRA
    processor/
    projector.pt             # chỉ E2/E3 và E4 dùng projector
    training_state.pt        # optimizer, scaler, RNG; chỉ giữ cho latest
    run.json
work/                        # predictions, manifest, điểm val, cấu hình E4
report/                      # backup evidence, ảnh, bảng kết quả
lab_results.zip              # chỉ chứa report/
```

Checkpoint mỗi 100 bước, gồm adapter **và projector**. Các checkpoint trước giữ adapter/projector;
optimizer/RNG chỉ giữ ở checkpoint mới nhất để giảm dung lượng. Chừa nhiều GB trống trên Drive
(E2/E3 có projector khoảng 109M tham số fp32 cộng optimizer; toàn bộ chuỗi có thể chiếm hơn 10 GB).

Nếu Colab mất session:

1. Chạy lại mục 0–2 để clone, mount, cài và chuẩn bị dataset.
2. Chạy lại cell 6 với **đúng các cấu hình/overrides cũ**. Notebook tự thêm `--resume` khi có `latest.json`.
3. Run hoàn tất được bỏ qua; run dở tiếp tục từ checkpoint gần nhất, khôi phục optimizer, scaler, RNG và vị trí batch.
4. Predictions đã ghi được giữ trên Drive; infer kiểm tra manifest rồi chỉ chạy những ảnh còn thiếu.

Có thể mất phần tính toán sau checkpoint gần nhất. Nếu bị ngắt trước checkpoint đầu tiên,
đổi thư mục run hoặc chuyển thư mục run dở sang tên khác trong Drive để bắt đầu lại.
Một dòng prediction bị ghi dở có thể làm JSONL không đọc được; giữ backup rồi bỏ riêng dòng cuối hỏng trước khi chạy lại.

Nếu thay cấu hình, dùng **run directory và prediction path mới**; không trộn kết quả cũ với model mới.
Notebook không tự pull code khi repo đã tồn tại; cập nhật repo Colab hoặc dùng checkout mới nếu code local thay đổi.

## 6. Khi gặp lỗi

| Lỗi | Xử lý |
|---|---|
| Không thấy `vlm_lab/runner.py` | Repo clone chưa có mã mới; push từ local hoặc upload toàn bộ source ZIP |
| Không có CUDA | Đổi runtime sang T4 GPU |
| torch/torchvision import lỗi | Kiểm tra phiên bản cặp thư viện của runtime; ưu tiên tạo runtime Colab mới. Không cài một bản torchvision bất kỳ với torch khác phiên bản |
| CUDA OOM khi train | Dùng batch 1, accumulation 8 cho **tất cả** setup; với run đã bắt đầu, tạo run mới để ghi nhận thay đổi. Checkpointing đã bật |
| CUDA OOM khi infer | Đặt `INFER_BATCH=1`, chạy lại cùng output path; số ảnh còn thiếu được tiếp tục |
| Loss hoặc gradient NaN/Inf | Dừng và kiểm tra mask/nhãn; giảm lr trong cấu hình riêng và thử val. Mã không âm thầm bỏ qua bước lỗi |
| Resume config mismatch | Dùng chính xác cấu hình lần đầu; thay đổi cấu hình thì tạo run mới |
| Expected 220/252/348 nhưng nhận số khác | Dừng, kiểm tra phiên bản và cây module; không xóa assert để chạy tiếp |
| Parse rate thấp | Đọc output val, kiểm tra nhãn có `<|im_end|>`; thử giới hạn sinh dài hơn trên val với output path mới |
| Drive đầy | Giữ checkpoint cuối cần dùng; chuyển/lưu trữ checkpoint lịch sử. Không xóa projector của run cần chấm |

CLI hỗ trợ train/infer/score/inspect/summarize/visualize; xem `python -m vlm_lab.runner --help`.
Ví dụ tiếp tục train đúng cấu hình mặc định:

```bash
python -m vlm_lab.runner train --config configs/e2.yaml \
  --run-dir /content/drive/MyDrive/road-vlm-day10/checkpoints/e2 --resume
```

## 7. Hoàn thành bài nộp

1. Lấy `lab_results.zip` từ Drive, giải nén `report/` về repo local.
2. Điền REPORT.md: họ tên, dataset_version, bảng kết quả thật và 10 câu. Tham khảo `report/results.md`, `data_checks.json`, raw predictions, log và ảnh minh họa.
3. Câu 3 có cell chẩn đoán F1 sai thang; Câu 9 cần chú thích/phân tích ảnh đúng/sai trong `report/figures/`. Bảng tự sinh không thay thế phân tích của bạn.
4. File → Download → Download .ipynb để lưu notebook đã chạy nếu muốn nộp kèm output. Kiểm tra output không chứa khóa bí mật.
5. Commit theo tiến độ với ít nhất 5 commit có ý nghĩa; không tạo commit giả chỉ để đủ số lượng. Không commit `data/`, `work/`, `.env`, checkpoints hoặc trọng số.
6. Chạy `python -m pytest -q` và `python -m pytest -q student_checks`.
7. Chạy `python scripts/check_submission.py`, xử lý từng FAIL rồi push repo công khai.

Ngưỡng rubric: E1 parse_valid_rate ≥ 0,95 và F1 tăng ≥ 0,10 so với E0.
Không đảm bảo các ngưỡng trước khi thực nghiệm; nếu chưa đạt, kiểm tra quy trình trên val và ghi nhận kết quả thật.

E5 **không bắt buộc**. Hiện có reward IoU và test trong `vlm_lab/reward.py`/`student_checks/test_reward.py`;
chưa có vòng GRPO, log hoặc evidence E5. Muốn bonus GRPO phải triển khai/chạy bổ sung và phân tích reward hacking.

Tham chiếu API: [Qwen3-VL Transformers](https://huggingface.co/docs/transformers/model_doc/qwen3_vl),
[PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization),
[PEFT checkpoints](https://huggingface.co/docs/peft/developer_guides/checkpoint).

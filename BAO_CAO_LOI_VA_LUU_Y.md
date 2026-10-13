# Báo cáo lỗi và lưu ý kỹ thuật — Lab Day 10

Nguyễn Đức Anh — MSSV 2A202602625. Kết quả thực nghiệm và phân tích chính ở [REPORT.md](REPORT.md).

| Vấn đề đã gặp | Nguyên nhân / tác động | Xử lý trong hồ sơ hiện tại |
|---|---|---|
| Cell kết thúc nhưng không hiện log | `subprocess.run` kế thừa stdout không hiển thị rõ trong Colab | `run_cmd` dùng PIPE, gộp stderr và stream từng dòng; lỗi vẫn làm cell dừng |
| `Unexpected quantized parameter: model.visual...` | Skip `visual` không khớp prefix đầy đủ của Transformers 5.19 | Dùng `model.visual`, giữ visual/projector/head float, xác nhận 252 Linear4bit LLM |
| Nhãn crack có `h=0` | `vlcsnap-00058` có box không có diện tích; JSON box đó không hợp lệ | Bỏ một box khỏi target train, giữ ảnh/nhãn khác và scorer gốc; thống kê trong `data_checks.json` |
| Tọa độ sai thang nhưng JSON đúng | Chuỗi parse được chưa đảm bảo box đúng vị trí | E1 chấm đúng F1=0,2616984402, chấm nhầm pixel F1=0; giữ chẩn đoán trong notebook và diagnostics |
| `torchao 0.10.0` không tương thích PEFT | PEFT 0.21.2 yêu cầu torchao ≥0.16 nếu thư viện này có mặt | Lab dùng bitsandbytes; môi trường mới gỡ riêng torchao cũ, không tùy tiện đổi torch/torchvision |
| `HFValidationError` khi resume | Thiếu file trọng số local khiến loader PEFT chuyển sang Hub; kiểu Path làm thông báo khó hiểu | Đọc adapter local trực tiếp, kiểm tra đủ file trước khi tải model; đường dẫn gọi PEFT là chuỗi |
| Checkpoint bước 100 chỉ có config/tokenizer | Metadata không đủ khôi phục trọng số hay optimizer | Kiểm tra cho thấy không resume được; E1 hoàn tất đã được train lại bước 1–300. Không khai phần bị mất là đã phục hồi |
| Rủi ro checkpoint trên Drive | Truyền file có thể bị ngắt/không khả dụng; không xác định được nguyên nhân mất file cũ chỉ từ log | Serialize trên local, copy sang `.incomplete`, đọc lại tensor trước khi cập nhật `latest`; giữ optimizer cũ để phục hồi. Không bảo đảm tuyệt đối nếu runtime/Drive lỗi |
| Tải/giải nén lâu | `work/` là symlink sang Drive; nhiều file nhỏ làm I/O chậm | Dùng `--work /content/dataset_cache`, giữ predictions/checkpoint trên Drive |
| Mất kết nối Colab | Có thể chỉ mất giao diện hoặc thật sự mất runtime; chưa có log đủ xác định mọi lần | Reconnect trước, kiểm tra trạng thái; runtime mới chuẩn bị lại dữ liệu/model. Chỉ resume nếu có đủ tensor và optimizer/RNG |
| Quên lưu projector E2/E3 | PEFT adapter không bao gồm projector được mở đầy đủ | Lưu/nạp `projector.pt` riêng, kiểm tra tất cả key projector |
| Test template mâu thuẫn bài nộp hoàn thiện | `test_report_template_has_student_markers_and_ten_questions` yêu cầu REPORT còn marker; checker lại yêu cầu bỏ marker | Giữ nguyên test gốc; ghi một failure dự kiến trong log toàn bộ tests, chạy các test phù hợp bản cuối và audit 10 câu/report riêng |

## Đối chiếu dữ liệu và đo lường

- Version **d10-21e4070301** được tái tạo độc lập bằng `scripts/prepare_data.py` nguyên bản từ dataset Kaggle; khớp evidence. Bản metadata ở `report/split_meta.json` là bản sao file sinh thật, không dựng từ số liệu báo cáo để vượt checker.
- Train/val/test có 1.422/276/311 ảnh và 3.515/639/583 box gốc. Chỉ train có một box bị loại khỏi target; val/test không có box bị loại theo audit.
- Mỗi evidence là test với 311 ảnh, `missing_predictions=0`. E4 100% dùng lại E3; không tính là lần train độc lập mới.
- Notebook hoàn tất xác nhận **L4**, batch 2, accumulation 4, seed 42, lr 0,0002, r=16, alpha 32, 300 bước. Không gán các thời gian này cho T4.
- `peak_vram_gb` thực tế là GiB do chia 1024³. Nó là bộ nhớ torch được cấp phát, không phải toàn bộ VRAM do `nvidia-smi` báo. Thời gian train khác thời gian tải model, suy luận hoặc tổng thời gian notebook.
- E1/E2/E3 mất 27,4008/28,3675/32,9733 phút train. E4 10% vẫn 300 bước, mất 32,3164 phút; ít dữ liệu không có nghĩa thời gian giảm còn 10%.
- Parse valid rate E1 99,68% và F1 tăng 0,200710 so với E0, vượt hai ngưỡng rubric. E3 F1=0,44 vẫn có 314 FP/330 FN, nên không coi model là auto-labeler đã đủ tin cậy.

## Giới hạn bằng chứng và tính trung thực

CI bootstrap theo ảnh không mô hình hóa tương quan frame trong cùng phiên và không đo biến thiên seed train. Không chạy đối chứng split ngẫu nhiên, không mask, projector fp16 hoặc nhiều seed. Phân tích nguyên nhân lỗi theo lớp là quan sát/hypothesis, không trình bày như kết quả ablation.

Năm E0 raw output lấy từ val smoke, không dùng chúng để khẳng định tần suất từng kiểu lỗi trên toàn test. Sáu ảnh mỗi setup là ảnh minh họa được chọn; thống kê tổng dùng đủ 311 ảnh.

E5 chỉ có reward và test, **chưa chạy GRPO**, không khai log/F1 GRPO hoặc bonus chưa có bằng chứng. Báo cáo có hỗ trợ viết mã/phân tích; người nộp cần hiểu và kiểm chứng trước khi vấn đáp.

## Phục hồi và chạy lại nghiệm thu

Trên clone mới, `data/` không được track. Để đối chiếu metadata đã xuất mà không commit dữ liệu:

```python
from pathlib import Path
import shutil
Path('data/splits').mkdir(parents=True, exist_ok=True)
shutil.copy2('report/split_meta.json', 'data/splits/meta.json')
```

Bản metadata lưu không thay thế việc tải/chia dữ liệu khi chạy thí nghiệm. Có thể tái tạo toàn bộ bằng công cụ gốc như hướng dẫn Colab.

Lệnh kiểm tra bản cuối:

```bash
python scripts/check_submission.py
python -m vlm_lab.submission_audit
python -m pytest -q tests student_checks -k "not test_report_template_has_student_markers_and_ten_questions"
```

Đây là ngoại lệ **test template** duy nhất, được ghi công khai; không sửa `scripts/`, `tests/`, `configs/` hay requirements của đề. Log chạy toàn bộ test vẫn được lưu để reviewer thấy failure dự kiến.

Checker gốc luôn in chi tiết “còn chuỗi” ở mục REPORT, kể cả khi nhãn là PASS, do thông báo detail
được gán cố định. Kết luận dựa vào trạng thái PASS/FAIL và audit kiểm tra trực tiếp nội dung REPORT;
không sửa công cụ đề chỉ để đổi câu thông báo.

Trên Windows PowerShell, đặt `$env:PYTHONIOENCODING="utf-8"` trước khi chạy. Nếu lệnh `python` của máy bị pyenv chặn, dùng executable Python đã cài; môi trường kiểm tra local dùng Python của Codex runtime.

Checkpoint, adapter, dataset, secrets và `work/` không được commit. Muốn nộp LMS, dùng URL repo public đã hoàn tất; giữ public cho đến khi có điểm chính thức.

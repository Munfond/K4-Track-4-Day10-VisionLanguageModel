# Checklist bài nộp Lab Day 10

Nguyễn Đức Anh — MSSV **2A202602625**. Dataset **d10-21e4070301**.

## Rà soát hồ sơ — 5/5

- [x] REPORT.md có họ tên, MSSV, dataset_version, bảng sáu dòng E0–E4 và đủ 10 câu trả lời bằng số liệu thực nghiệm.
- [x] REPORT.md không còn marker cần điền của template.
- [x] Có **18 PNG** dự đoán đúng/sai, box GT/prediction, caption TP/FP/FN; sáu ảnh E3 được nhúng và phân tích trong báo cáo.
- [x] Không track `data/`, `work/`, `.env`, checkpoint hoặc trọng số `.safetensors`/`.bin`/`.pt`/`.ckpt`.
- [x] Lịch sử Git có ít nhất 5 commit có ý nghĩa; bản notebook chạy thật được người nộp push ở commit `da2e5e0`.

## Thí nghiệm và tính nhất quán

- [x] E0–E3 đủ file, cùng version, split test, 311 ảnh và không thiếu predictions.
- [x] E1–E3 có trainable_params, peak_vram_gb và train_minutes dương; khớp run log notebook.
- [x] E1 parse rate **0,9967845659 ≥ 0,95**; F1 tăng **0,2007100070 ≥ 0,10** so với E0.
- [x] E4 hai giá trị 10%/100%, nền E3 chọn trên val, giữ cấu hình khác và 300 bước. Evidence 100% trùng E3 ngoài tên run.
- [x] Metadata tái tạo bằng công cụ gốc khớp version và số liệu, được giữ ở `report/split_meta.json`; bản đối chiếu local ở `data/splits/meta.json` bị Git ignore.
- [x] Chẩn đoán sai tọa độ đủ 311 ảnh ghi trong notebook: E1 F1 đúng 0,2616984402, sai pixel 0.
- [x] Scripts, tests, configs và requirements không thay đổi so với commit đề `c2f3997`.
- [x] Notebook có output GPU L4, mask loss/220 token, E0 raw, train/val/test, E4 và xuất kết quả; đã kiểm tra không có API key dạng literal trong source/output.

## Nghiệm thu

- [x] Chạy `scripts/check_submission.py` nguyên bản và yêu cầu **0 FAIL, 0 WARN**.
- [x] Audit bổ sung khớp số liệu bảng, đếm TP/FP/FN, config/log, schema và ngưỡng rubric.
- [x] Đối chiếu lại **18 caption** ảnh với GT của split nguyên bản và scorer; đọc được 18 PNG.
- [x] Chạy test gốc và student checks, lưu cả log toàn bộ test và log áp dụng cho bản báo cáo hoàn thiện.
- [x] Ghi rõ test template duy nhất mâu thuẫn với REPORT hoàn thiện; không sửa/xóa test hoặc làm giả kết quả. Xem `BAO_CAO_LOI_VA_LUU_Y.md`.

Log nghiệm thu được lưu trong `report/verification/`; audit máy đọc ở `report/submission_audit.json`. Các log là snapshot lúc kiểm tra, số commit có thể tăng sau khi commit hồ sơ.

## GitHub và LMS

- [x] Repo `Munfond/K4-Track-4-Day10-VisionLanguageModel` được kiểm tra là **Public** qua GitHub API ngày 14/10/2026.
- [ ] Người nộp đọc lại REPORT, đặc biệt cách đo thời gian/VRAM, hạn chế CI và dữ liệu lỗi, để xác nhận hiểu và chuẩn bị vấn đáp.
- [ ] Dán URL repo vào ô nộp **Day 10 trên VinUni LMS**, xác nhận gửi bài và kiểm tra trạng thái nộp. Chưa có URL trang LMS nên chưa xác nhận đã nộp.
- [ ] Giữ repo Public đến khi điểm chính thức được công bố.

URL nộp: https://github.com/Munfond/K4-Track-4-Day10-VisionLanguageModel

E5 GRPO không thuộc phần bắt buộc và chưa được chạy; không đánh dấu bonus GRPO là đã hoàn thành.

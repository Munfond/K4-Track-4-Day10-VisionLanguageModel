# Báo cáo Lab Day 10 — Grounding hư hại mặt đường bằng Qwen3-VL-4B

Họ tên: **Nguyễn Đức Anh**

MSSV: **2A202602625**

`dataset_version`: **d10-21e4070301**

Model: `Qwen/Qwen3-VL-4B-Instruct`. GPU của các run hoàn tất: **NVIDIA L4 trên Colab**, được xác nhận trong output mục 0 của notebook.

## Bảng tổng hợp kết quả thật

Nguồn: [evidence](report/evidence/), chấm trên cùng 311 ảnh test bằng `scripts/score.py` nguyên bản, IoU ≥ 0,5 và ghép cùng lớp. F1 là micro-F1; CI95 là bootstrap theo ảnh. Bảng làm tròn, JSON giữ độ chính xác gốc.

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GiB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| E0 | Zero-shot | 0 | 3,7794 | 32,1554* | 0,0610 | [0,0406; 0,0817] | 1,0900 | 0,1220 | 0,0086 | 0,1059 |
| E1 | LoRA LLM | 33.030.144 | 6,6774 | 27,4008 | 0,2617 | [0,2247; 0,3023] | 1,3505 | 0,1696 | 0,1967 | 0,5097 |
| E2 | E1 + projector đầy đủ | 142.135.296 | 8,3719 | 28,3675 | 0,3495 | [0,3103; 0,3921] | 1,1318 | 0,2130 | 0,2839 | 0,6087 |
| E3 | E2 + LoRA ViT | 148.426.752 | 8,5501 | 32,9733 | 0,4400 | [0,4009; 0,4838] | 1,0096 | 0,3077 | 0,3799 | 0,6829 |
| E4 — 10% | E3, 142 ảnh train | 148.426.752 | 7,9501 | 32,3164 | 0,2890 | [0,2564; 0,3236] | 1,7235 | 0,1469 | 0,2199 | 0,5931 |
| E4 — 100% | Dùng lại E3, 1.422 ảnh train | 148.426.752 | 8,5501 | 32,9733 | 0,4400 | [0,4009; 0,4838] | 1,0096 | 0,3077 | 0,3799 | 0,6829 |

\* E0 không train: `train_minutes=0`; 32,1554 phút và VRAM E0 là **suy luận test**, không so trực tiếp với thời gian train E1–E3. VRAM đo bằng `max_memory_allocated / 1024**3`, nên đơn vị thực là GiB dù key đề đặt là `peak_vram_gb`. Thời gian train không bao gồm tải/nạp model, có thể gồm ghi checkpoint trước đó nhưng không gồm lần ghi checkpoint cuối. E4 100% dùng lại E3, không cộng thành một lần train mới.

Các run giữ `r=16`, alpha 32, dropout 0,05, lr 0,0002, batch 2, accumulation 4, 300 bước, seed 42 và tọa độ `norm1000`. [Cấu hình và output trích notebook](report/diagnostics/notebook_run_facts.json) xác nhận E1–E3 hoàn tất bước 300.

## 10 câu hỏi

### Câu 1 — Kiến trúc và các khối được cập nhật

ViT mã hóa ảnh thành đặc trưng; projector chuyển đặc trưng sang không gian hidden của LLM; LLM sinh JSON nhãn/box. DeepStack đưa thêm đặc trưng trung gian ViT qua các merger riêng vào LLM. Ảnh 640×360 trong kiểm tra sinh **220 token ảnh**.

[Đếm độc lập trên kiến trúc thật](report/diagnostics/architecture_parameter_counts.json), không phải trọng số của run train, cho tổng duy nhất **4.437.815.808**; visual gồm projector **415.347.712**; LLM gồm embedding **4.022.468.096**. Merger chính có **27.271.680**, DeepStack merger tổng **81.833.472**, cộng **109.105.152** tham số projector. `lm_head` có 388.956.160 tham số dùng chung embedding, không cộng lần nữa. Visual ngoài projector là 306.242.560 tham số.

E1 học LoRA LLM trên 252 module attention/MLP: **33.030.144** tham số; visual/projector đóng băng. E2 mở thêm toàn bộ projector, tổng **142.135.296**. E3 thêm LoRA ở 96 module ViT, tổng 348 target module và **148.426.752** tham số học. Chênh lệch E3−E2 **6.291.456** là adapter ViT, không phải mở đầy đủ ViT.

### Câu 2 — E0 và lỗi quan sát được

E0 có `parse_valid_rate=0,5176848875`: **161/311 ảnh** parse được, 150 ảnh không parse được. F1 **0,0609884332**, TP=29, FP=339, FN=554; F1 crack thấp nhất **0,0085836910**.

Trong năm output val được in ở mục 3, hai kiểu lỗi nổi bật là **lặp/bịa nhiều box**, có box rất rộng, và **JSON chưa kết thúc**. Raw của `vlcsnap-2025-02-18-18h30m17s408` lặp bốn lần box crack, rồi dừng ở trường box đang dở:

```text
[{"label": "crack", "bbox_2d": [0, 370, 1000, 700], "confidence": 0.7}, {"label": "crack", "bbox_2d": [0, 370, 1000, 700], "confidence": 0.7}, {"label": "crack", "bbox_2d": [0, 370, 1000, 700], "confidence": 0.7}, {"label": "crack", "bbox_2d": [0, 370, 1000, 700], "confidence": 0.7}, {"label": "manhole", "bbox_2d": [350, 286, 367, 300], "confidence": 0.7}, {"label": "manhole", "bbox_2d": [350, 286, 367, 300], "confidence": 0.7}, {"label": "manhole", "bbox_2d":
```

Output parse được ở `vlcsnap-2025-02-18-18h30m25s530` vẫn lặp box:

```json
[{"label":"manhole","bbox_2d":[607,515,675,545]},{"label":"crack","bbox_2d":[0,375,1000,999]},{"label":"crack","bbox_2d":[0,375,1000,999]}]
```

Box crack phủ khoảng 62,4% ảnh. Trường `confidence` thừa so với prompt nhưng scorer bỏ qua trường thừa, nên riêng nó không làm JSON sai. Chuỗi chưa kết thúc phù hợp với sinh lặp và giới hạn 256 token; chưa đủ log để quy mọi lỗi cho giới hạn token. Năm ví dụ val không xác lập thứ hạng tần suất lỗi trên toàn test. [Raw và nguồn notebook](report/diagnostics/notebook_run_facts.json) giữ đủ các ví dụ.

### Câu 3 — Thang 0–1000 và kiểm chứng sai thang

Tọa độ chuẩn hóa biểu diễn box độc lập với số pixel. Với ảnh 640×360: `x_pixel=x_norm*640/1000`, `y_pixel=y_norm*360/1000`; x và y có hệ số khác nhau.

Chấm **cùng output E1 trên đủ 311 ảnh test**: thang đúng cho **F1=0,2616984402**; `--coord pixel` sai cho **F1=0,0**, CI95=[0,0], mean IoU=0, trong khi parse rate vẫn xấp xỉ 0,997. Output mục 9 nằm trong [bằng chứng chẩn đoán](report/diagnostics/notebook_run_facts.json). Đây là kiểm tra chấm sai, không phải E4 tọa độ.

Ví dụ `[289,661,398,728]` phải đổi thành khoảng `[184,96; 237,96; 254,72; 262,08]` pixel. Nếu coi y=661–728 là pixel trên ảnh cao 360, box nằm ngoài ảnh. Kiểm tra bổ sung dùng sáu ảnh minh họa E1 với scorer nguyên bản cho F1 đúng **0,5**, sai **0,0**; [kết quả sáu ảnh](report/diagnostics/e1_six_examples_coord_check.json) ghi rõ phạm vi, không thay số liệu toàn test.

### Câu 4 — Mask loss

Loss chỉ tính trên đáp án assistant để học sinh JSON từ ảnh và yêu cầu. Token prompt, token ảnh và padding có label **−100**. Nếu tính cả prompt, mục tiêu học bao gồm tái tạo token yêu cầu/placeholder đã được cung cấp, phân tán tín hiệu so với nhiệm vụ sinh đáp án. Token ảnh là placeholder nối với đặc trưng visual, không phải mục tiêu tái tạo pixel.

Mục 2 xác nhận prompt là tiền tố chuỗi đầy đủ, có **220 token ảnh**, và decode các label khác −100 ra JSON rồi `<|im_end|>`. Collator kiểm tra padding −100 và nối pixel/grid không pad. E1 có loss bước 1 **1,2898**, bước 300 **0,5544**, parse rate **310/311=99,6785%**, F1 tăng **0,2007100070** so với E0. Kết quả phù hợp với học đáp án nhưng không thay kiểm tra mask; chưa chạy đối chứng không mask nên không quy toàn bộ mức tăng cho riêng mask loss.

### Câu 5 — Lượng tử hóa và projector float32

Base weight 4-bit của bitsandbytes là biểu diễn nén dùng cho model đóng băng, không phải trọng số float có thể cập nhật đầy đủ bằng AdamW theo setup này. E2/E3 cần học **109.105.152** tham số projector nên loại `model.visual` và `lm_head` khỏi lượng tử hóa; 252 Linear LLM dùng NF4 4-bit.

Skip list phải là **`model.visual`** ở Transformers 5.19; tên `visual` đã gây `Unexpected quantized parameter: model.visual.blocks.0.attn.qkv.weight`. Bản sửa kiểm tra visual/head còn float và số Linear4bit LLM đúng 252.

Projector chuyển sang **float32** để giữ cập nhật/gradient chính xác hơn và tương thích `GradScaler.unscale_`; mọi tham số trainable, kể cả adapter, cũng giữ fp32. Forward vẫn autocast fp16. Scaling giảm underflow ở gradient trung gian; master weight fp32 giảm mất cập nhật nhỏ nhưng không bảo đảm hết NaN. E2/E3 hoàn tất với VRAM **8,3719/8,5501 GiB**; chưa có đối chứng projector fp16 để định lượng độ ổn định.

### Câu 6 — Split theo phiên quay

Frame cùng phiên gần giống nhau; chia ngẫu nhiên có thể đưa gần như cùng cảnh vào train và test, làm F1 phản ánh ghi nhớ cảnh hơn là tổng quát sang phiên khác. Công cụ gom phiên từ tên file; ảnh `vlcsnap-<số>` còn gom theo nhóm chỉ số. Kiểm tra xác nhận khóa phiên không giao nhau giữa split.

| Split | Ảnh | Box pothole | Box crack | Box manhole | Tổng box |
|---|---:|---:|---:|---:|---:|
| Train | 1.422 | 1.074 | 1.750 | 691 | 3.515 |
| Val | 276 | 66 | 470 | 103 | 639 |
| Test | 311 | 121 | 299 | 163 | 583 |

Nguồn: [data_checks.json](report/data_checks.json), [metadata nguyên bản tái tạo](report/split_meta.json). Version tái tạo khớp `d10-21e4070301`. Val chỉ có 66 pothole, nên cần xem từng lớp. Không chạy split ngẫu nhiên làm đối chứng và không báo mức tăng F1 giả định như số đo thật.

### Câu 7 — Chi phí và lợi ích E1/E2/E3

E2 tăng F1 từ **0,2617 lên 0,3495**, thêm **8,78 điểm phần trăm**; VRAM thêm **1,6945 GiB**, thời gian thêm **0,9668 phút**. FP/ảnh giảm **1,3505 → 1,1318**. Cận trên CI E1 0,3023 thấp hơn cận dưới E2 0,3103.

E3 tăng thêm **9,05 điểm phần trăm** lên **0,4400**; VRAM chỉ thêm **0,1782 GiB**, thời gian thêm **4,6057 phút**. FP/ảnh giảm còn **1,0096**; cả ba F1 lớp đều tăng. Cận dưới E3 0,4009 cao hơn cận trên E2 0,3921.

Trên L4 đã đo, **E3 đáng chọn nhất nếu ưu tiên chất lượng**, với phần VRAM tăng thêm nhỏ so với E2. E2 là lựa chọn nếu ưu tiên thời gian; E1 tiết kiệm VRAM nhất. CI không chồng lấn ủng hộ thứ tự điểm trên test này, nhưng không thay kiểm định chênh lệch ghép cặp hoặc đảm bảo thứ tự trên mọi dữ liệu. Chưa chạy nhiều seed/paired bootstrap.

### Câu 8 — Giới hạn 311 ảnh test và CI95

Độ rộng CI E0/E1/E2/E3 là **0,0410; 0,0776; 0,0819; 0,0828**. E3 có CI **[0,4009; 0,4838]**: chỉ báo F1=0,44 che mức bất định khoảng 4 điểm phần trăm mỗi phía. CI mô tả thủ tục lấy mẫu, không phải xác suất F1 thật nằm trong một khoảng đã tính.

Scorer bootstrap **1.000 lần theo ảnh**, seed 0. Frame cùng phiên có thể tương quan dù train/test không trùng phiên; bootstrap ảnh chưa mô hình hóa tương quan này và có thể đánh giá thấp bất định. Giữ scorer nguyên bản, ghi nhận hạn chế. CI cũng chưa bao gồm biến thiên seed train, tuyến đường, thời tiết/camera khác. Chênh lệch nhỏ cần thêm dữ liệu, multiple-seed hoặc bootstrap theo phiên/ghép cặp để đánh giá chắc hơn.

### Câu 9 — Lỗi từng lớp và ảnh minh họa

Ở E3, **pothole kém nhất**: F1 **0,3077**, TP=38, FP=88, FN=83, recall 0,3140. Crack có F1 0,3799 nhưng FP=200, FN=182; manhole tốt nhất, F1 0,6829. Test có 121 box pothole, ít hơn crack 299/manhole 163; bóng và cấu trúc mặt đường có thể gây nhầm. Manhole có viền/cấu trúc rõ trong nhiều ảnh đã xem. Đây là giải thích từ quan sát, chưa phải thống kê kích thước từng lớp hoặc kiểm chứng nguyên nhân.

Repo có **18 ảnh đúng/sai** của E1–E3. Sáu ảnh E3 dưới đây: xanh là ground truth, đỏ là prediction. Mẫu được chọn để minh họa, không dùng tỷ lệ đúng của sáu ảnh làm điểm test.

| Ảnh E3 | TP / FP / FN | Nhận xét |
|---|---|---|
| [vlcsnap-00200](report/figures/e3/example_1.png) | 2 / 0 / 0 | Crack/manhole đúng; E1 cùng ảnh có 1 TP, 1 FP, 1 FN. |
| [vlcsnap-00204](report/figures/e3/example_2.png) | 1 / 0 / 0 | Manhole khớp ground truth. |
| [vlcsnap-00205](report/figures/e3/example_3.png) | 2 / 0 / 0 | Hai vùng crack ghép đúng. |
| [vlcsnap-00201](report/figures/e3/example_4.png) | 0 / 1 / 2 | Crack lệch/co hẹp, IoU chưa đạt 0,5; manhole bị bỏ sót. |
| [vlcsnap-00202](report/figures/e3/example_5.png) | 1 / 1 / 0 | Manhole đúng nhưng thêm một pothole không có GT. |
| [vlcsnap-00203](report/figures/e3/example_6.png) | 0 / 1 / 1 | Chỉ khoanh một phần crack rộng; box thành FP và GT thành FN. |

![Crack và manhole đúng](report/figures/e3/example_1.png)
![Manhole đúng](report/figures/e3/example_2.png)
![Hai crack đúng](report/figures/e3/example_3.png)
![Crack lệch và manhole bị bỏ sót](report/figures/e3/example_4.png)
![Pothole dự đoán thừa](report/figures/e3/example_5.png)
![Chỉ khoanh một phần crack](report/figures/e3/example_6.png)

Raw và ví dụ E1/E2: [index E1](report/figures/e1/index.md), [index E2](report/figures/e2/index.md), [index E3](report/figures/e3/index.md).

### Câu 10 — E4 lượng dữ liệu 10%/100%

Chỉ đổi **`train_fraction`**, dùng E3 vì F1 val E1/E2/E3 là **0,2133 / 0,2995 / 0,3470**. Chọn trên val, không dùng test. 10% là **142 ảnh** lấy mẫu seed 42; 100% là 1.422 ảnh. Giữ LoRA/projector, prompt, thang tọa độ, seed, lr, batch/accumulation và 300 bước. 100% dùng lại predictions E3, chấm thành evidence E4 riêng.

F1 10% **0,2890 [0,2564; 0,3236]**, so với 100% **0,4400 [0,4009; 0,4838]**, giảm **15,10 điểm phần trăm**. FP/ảnh tăng **1,0096 → 1,7235** khi dùng ít dữ liệu. Tham số học bằng nhau; thời gian **32,3164 vs 32,9733 phút** gần nhau vì cùng 300 bước, không cùng số epoch. Ngân sách danh nghĩa 2×4×300=2.400 lượt ảnh; tập nhỏ lặp nhiều hơn, batch cuối epoch có thể nhỏ hơn 2.

Loss 10% giảm **1,4616 → 0,0205** nhưng test F1 thấp, phù hợp dấu hiệu học thuộc tập nhỏ. Không dùng loss thấp để tuyên bố tổng quát tốt. CI không chồng lấn ủng hộ lợi ích dữ liệu đầy đủ trên test này; vẫn chỉ là một subset seed, một seed train và 311 ảnh có thể tương quan. Không suy ra quy luật cho mọi mức dữ liệu từ hai mức đã chạy.

## Nhìn lại

Train có một nhãn crack `vlcsnap-00058`: `[0.639062,0.786111,0.003125,0.0]`, chiều cao 0. Mã bỏ **một box** khỏi target SFT nhưng giữ ảnh/nhãn khác, split và scorer gốc. Có 3.515 box train gốc, còn 3.514 target có diện tích. Val/test không có box bị bỏ; box lỗi này không tạo FN trực tiếp trong test hiện tại. Không loại mẫu để làm đẹp điểm chấm.

Các lỗi đã gặp: skip lượng tử hóa sai đường dẫn, torchao cũ, runtime mất kết nối và checkpoint thiếu tensor. E1 thiếu checkpoint đã được chạy lại; run hoàn tất trong notebook chạy bước 1–300, không coi các bước đã mất là đã phục hồi. Mã hiện tại đọc lại checkpoint trước khi cập nhật latest. [BAO_CAO_LOI_VA_LUU_Y.md](BAO_CAO_LOI_VA_LUU_Y.md) ghi bẫy và cách xử lý.

**E5 chưa chạy GRPO.** Có reward IoU và hai test reward đạt; không có evidence GRPO và không tuyên bố bonus chạy RL. Chưa thực nghiệm bỏ mask, projector fp16, nhiều seed hoặc dữ liệu khác miền. F1 E3=0,44 và khoảng một FP/ảnh cho thấy auto-labeler vẫn cần kiểm tra thủ công.

Báo cáo được hoàn thiện với hỗ trợ viết mã/phân tích; số liệu/output lấy từ notebook thực nghiệm của người nộp. Người nộp cần đọc, kiểm chứng và hiểu kết luận để chuẩn bị vấn đáp.

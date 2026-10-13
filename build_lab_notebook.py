"""Rebuild the completed notebook without touching the instructor's scripts/."""
import json
from pathlib import Path

cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(keepends=True)})


def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": source.strip().splitlines(keepends=True)})


md("""
# Lab Day 10 — Qwen3-VL-4B: triển khai E0–E4
Chạy các cell **theo thứ tự** trên Colab T4. Mã của người làm nằm trong `vlm_lab/`;
giữ nguyên `scripts/` và `tests/`. Xem `COLAB.md` để biết cách tiếp tục khi mất session.
Đầu ra và số liệu chỉ được tạo từ những lần chạy thật. E5 là bonus: đã có reward + test,
chưa tích hợp vòng GRPO. Bạn cần đọc hiểu mã, kiểm chứng và tự viết 10 câu trong REPORT.md.
""")
md("""
## 0. Môi trường (Colab T4)
Runtime → Change runtime type → **T4 GPU**. Sửa `REPO_URL` thành repo của bạn chứa mã này.
Nếu dùng ZIP, giải nén repo vào `/content/road-vlm` và đặt `REPO_URL = ""`.
Google Drive giữ checkpoint và predictions; dữ liệu và model cache nằm trên ổ Colab cho nhanh.
""")
code('''
from google.colab import drive
drive.mount('/content/drive')
import os, sys, subprocess, shutil, json
from pathlib import Path

def run_cmd(*args, check=True):
    command = [str(a) for a in args]
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    print("Running:", " ".join(command), flush=True)
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", bufsize=1,
                          env=child_env) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        returncode = process.wait()
    if check and returncode:
        raise subprocess.CalledProcessError(returncode, command)
    return subprocess.CompletedProcess(command, returncode)

REPO_URL = "https://github.com/Munfond/K4-Track-4-Day10-VisionLanguageModel.git"
REPO_DIR = Path("/content/road-vlm")
ARTIFACT_DIR = Path("/content/drive/MyDrive/road-vlm-day10")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
if not REPO_DIR.exists():
    assert REPO_URL, "Giải nén ZIP vào /content/road-vlm trước."
    run_cmd("git", "clone", REPO_URL, str(REPO_DIR))
os.chdir(REPO_DIR)
assert (REPO_DIR / "vlm_lab/runner.py").exists(), "Repo Colab chưa có mã mới: push mã hoặc upload ZIP."
# Khôi phục bằng chứng đã lưu nếu đây là session mới.
if (ARTIFACT_DIR / "report").exists():
    shutil.copytree(ARTIFACT_DIR / "report", REPO_DIR / "report", dirs_exist_ok=True)
# Predictions ghi trực tiếp vào Drive để có thể tiếp tục batch còn thiếu.
(ARTIFACT_DIR / "work").mkdir(exist_ok=True)
work = REPO_DIR / "work"
if not work.exists():
    work.symlink_to(ARTIFACT_DIR / "work", target_is_directory=True)
else:
    assert work.is_symlink() and work.resolve() == (ARTIFACT_DIR / "work").resolve(), "work đã tồn tại; chọn checkout Colab mới."
CHECKPOINT_ROOT = ARTIFACT_DIR / "checkpoints"
CHECKPOINT_ROOT.mkdir(exist_ok=True)

def lab(*args):
    run_cmd(sys.executable, "-m", "vlm_lab.runner", *args)

def backup_reports():
    shutil.copytree(REPO_DIR / "report", ARTIFACT_DIR / "report", dirs_exist_ok=True)

run_cmd("nvidia-smi", "-L")
run_cmd(sys.executable, "-m", "pip", "install", "-r", "requirements-colab.txt", "pytest")
# Colab có thể cài sẵn torchao quá cũ cho PEFT. Lab dùng bitsandbytes, không dùng torchao.
from importlib.metadata import version, PackageNotFoundError
from packaging.version import Version
try:
    torchao_version = version("torchao")
except PackageNotFoundError:
    torchao_version = None
if torchao_version is not None and Version(torchao_version) < Version("0.16.0"):
    print(f"Removing incompatible optional torchao {torchao_version}; lab uses bitsandbytes.")
    run_cmd(sys.executable, "-m", "pip", "uninstall", "-y", "torchao")
# Colab có thể có xung đột từ các package không dùng trong lab; vẫn in để kiểm tra.
run_cmd(sys.executable, "-m", "pip", "check", check=False)
run_cmd(sys.executable, "-c", "import torch, torchvision, transformers, peft, bitsandbytes; "
        "assert torch.cuda.is_available(); print('GPU and lab imports: OK')")
''')
md("""
Các lệnh GPU chạy bằng subprocess nên không cần import torch/model vào kernel notebook.
Nếu cài đặt báo xung đột torch/torchvision, xem COLAB.md trước khi chạy tiếp.
""")
code('''
run_cmd(sys.executable, "-m", "pytest", "-q")
run_cmd(sys.executable, "-m", "pytest", "-q", "student_checks")
''')
md("""
## 1. Dữ liệu: tải và chia split
Chọn `DATA_MODE`: `kaggle`, `zip`, hoặc `source`. Kaggle: tạo Colab Secrets
`KAGGLE_USERNAME`, `KAGGLE_KEY`, bật quyền truy cập cho notebook. Không ghi khóa vào cell.
""")
code('''
DATA_MODE = "kaggle"
DATA_PATH = "/content/drive/MyDrive/dataset.zip"  # chỉ dùng cho zip/source
DATA_CACHE = "/content/dataset_cache"  # tải/giải nén trên ổ Colab, không dùng work/ trên Drive
if DATA_MODE == "kaggle":
    from google.colab import userdata
    os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
    run_cmd(sys.executable, "scripts/prepare_data.py", "--kaggle", "--work", DATA_CACHE)
elif DATA_MODE in ("zip", "source"):
    run_cmd(sys.executable, "scripts/prepare_data.py", "--"+DATA_MODE, DATA_PATH, "--work", DATA_CACHE)
else:
    raise ValueError(DATA_MODE)
''')
md("""
## 2. Nhãn: YOLO → JSON thang 0–1000
Cell này kiểm tra split, chuyển đổi qua lại tọa độ, prefix chat template, mask loss,
220 token ảnh và collator. Xem 6 ảnh: xanh là nhãn gốc, đỏ là nhãn đã chuyển đổi.
""")
code('''
lab("inspect")
from IPython.display import display, Image as DisplayImage, Markdown
for image in sorted(Path("report/figures/data").glob("*.png")):
    display(DisplayImage(filename=str(image)))
backup_reports()
''')
md("""
## 3. E0: zero-shot
Chạy thử 8 ảnh val, đọc output thô; sau đó chạy đủ val và test.
Không chỉnh prompt/hyperparameter dựa trên test. Predictions có manifest để tránh trộn các lần chạy.
""")
code('''
INFER_BATCH = 4  # Nếu OOM, đổi thành 1; không thay đổi dự đoán greedy theo thiết kế.
lab("infer", "--split", "val", "--limit", 8, "--batch-size", INFER_BATCH, "--out", "work/e0_smoke.jsonl")
from scripts.labcommon import read_jsonl
for row in read_jsonl("work/e0_smoke.jsonl")[:5]:
    print(row["image_id"], row["raw"], "\\n")
''')
code('''
Path("work/validation").mkdir(parents=True, exist_ok=True)
lab("infer", "--split", "val", "--batch-size", INFER_BATCH, "--out", "work/e0_val.jsonl")
lab("score", "--pred", "work/e0_val.jsonl", "--name", "E0-val", "--out", "work/validation/e0.json")
lab("infer", "--split", "test", "--batch-size", INFER_BATCH, "--out", "work/e0.jsonl")
lab("score", "--pred", "work/e0.jsonl", "--name", "E0", "--out", "report/evidence/e0.json")
backup_reports()
''')
md("""
## 4. Dựng dữ liệu huấn luyện: chat template và mask loss
Các hàm nằm trong `vlm_lab/modeling.py`: `encode_example`, `collate_examples`.
Chạy smoke test 2 bước E1 để kiểm tra forward/backward/save/load trước lần train thật.
Smoke test có thư mục riêng và không được nộp làm bằng chứng E1.
""")
code('''
smoke_dir = CHECKPOINT_ROOT / "smoke_e1"
smoke_args = ["train", "--config", "configs/e1.yaml", "--run-dir", smoke_dir,
              "--max-steps", 2, "--save-every", 1]
if (smoke_dir / "latest.json").exists():
    smoke_args.append("--resume")
lab(*smoke_args)
lab("infer", "--checkpoint", smoke_dir, "--split", "val", "--limit", 8,
    "--batch-size", INFER_BATCH, "--out", "work/e1_smoke.jsonl")
''')
md("""
## 5. Setup LoRA / đóng băng cho E1, E2, E3
E1: 252 module LLM. E2: cùng LoRA + projector fp32 (~109M). E3: 348 module LLM+ViT + projector.
`apply_setup` xác nhận module và lưu `modules.json`. Mọi tham số được học ở fp32 để GradScaler hoạt động.
""")
code('''
stats = json.loads((smoke_dir / "modules.json").read_text())
print("E1 trainable params:", stats["trainable_params"])
print("Matched modules:", len(stats["matched_modules"]))
print("First targets:", stats["matched_modules"][:10])
''')
md("""
## 6. Vòng huấn luyện
Chạy E1 → E2 → E3, mỗi lần khởi tạo từ model gốc. Mặc định 300 bước như configs.
Checkpoint mỗi 100 bước (adapter, projector nếu có, optimizer, scaler, RNG).
Để xử lý OOM, đặt `TRAIN_OVERRIDES = ["--batch-size", 1, "--grad-accum", 8]`
**trước lần chạy đầu**, dùng cùng lựa chọn cho E1–E3 và E4 để so sánh công bằng.
Sau khi mất session: chạy lại các bước chuẩn bị rồi cell này với cùng cấu hình; run hoàn tất được bỏ qua.
""")
code('''
TRAIN_OVERRIDES = []
# Có thể chạy từng setup bằng RUNS = ["e1"], rồi ["e2"], rồi ["e3"].
RUNS = ["e1", "e2", "e3"]
for key in RUNS:
    directory = CHECKPOINT_ROOT / key
    args = ["train", "--config", f"configs/{key}.yaml", "--run-dir", directory, *TRAIN_OVERRIDES]
    if (directory / "latest.json").exists():
        args.append("--resume")
    lab(*args)
    lab("infer", "--checkpoint", directory, "--split", "val", "--batch-size", INFER_BATCH,
        "--out", f"work/{key}_val.jsonl")
    lab("score", "--pred", f"work/{key}_val.jsonl", "--name", key.upper()+"-val",
        "--out", f"work/validation/{key}.json")
    print(json.loads((directory / "run.json").read_text()))
''')
md("""
## 7. Suy luận và chấm E1–E3
Chỉ chạy sau khi train hoàn tất và đã kiểm tra val. Chấm đủ 311 ảnh test cho mỗi setup.
Scorer lấy metadata trực tiếp từ run.json. Cell tạo bảng kết quả và 6 ảnh minh họa đúng/sai.
""")
code('''
for key in RUNS:
    lab("infer", "--checkpoint", CHECKPOINT_ROOT / key, "--split", "test",
        "--batch-size", INFER_BATCH, "--out", f"work/{key}.jsonl")
    lab("score", "--pred", f"work/{key}.jsonl", "--name", key.upper(),
        "--out", f"report/evidence/{key}.json")
    lab("visualize", "--pred", f"work/{key}.jsonl", "--out", f"report/figures/{key}")
    backup_reports()
lab("summarize")
display(Markdown(Path("report/results.md").read_text()))
''')
md("""
## 8. E4 (ablation tự chọn) và E5 (tuỳ chọn, GRPO)
Mặc định E4 là **data_size**: 10% vs 100% train, giữ các cấu hình khác của setup tốt nhất trên val.
Kết quả 100% được dùng lại từ run nền; chạy thêm 10%. Có thể đổi `E4_VARIABLE` thành `rank`
(4 vs 64) hoặc `coord_scale` (norm1000 vs pixel). Với pixel, prompt/nhãn/scorer tự đổi đồng bộ.
Không đổi lựa chọn E4 dựa trên điểm test. Rank 64 trên E3 có thể vượt VRAM T4.
""")
code('''
import copy, yaml
E4_VARIABLE = "data_size"
keys = ["e1", "e2", "e3"]
val_scores = {k: json.loads(Path(f"work/validation/{k}.json").read_text())["micro"]["f1"] for k in keys}
BEST = max(keys, key=lambda k: val_scores[k])
base_run = json.loads((CHECKPOINT_ROOT / BEST / "run.json").read_text())
base_cfg = base_run["config"]
print("Base chosen on val:", BEST, val_scores)
values = {"rank": [4, 64], "data_size": [0.1, 1.0], "coord_scale": ["norm1000", "pixel"]}[E4_VARIABLE]
for value in values:
    cfg = copy.deepcopy(base_cfg)
    if E4_VARIABLE == "rank":
        cfg["lora"].update(r=value, alpha=2*value)
    elif E4_VARIABLE == "data_size":
        cfg["train_fraction"] = value
    else:
        cfg["coord"] = value
    key = f"e4_{E4_VARIABLE}_{value}"
    cfg_path = Path(f"work/{key}.yaml")
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    reuse_base = cfg == base_cfg
    if reuse_base:
        directory = CHECKPOINT_ROOT / BEST
        pred_file = f"work/{BEST}.jsonl"
    else:
        directory = CHECKPOINT_ROOT / key
        args = ["train", "--config", cfg_path, "--run-dir", directory]
        if (directory / "latest.json").exists():
            args.append("--resume")
        lab(*args)
        lab("infer", "--checkpoint", directory, "--split", "val", "--batch-size", INFER_BATCH,
            "--out", f"work/{key}_val.jsonl")
        lab("score", "--pred", f"work/{key}_val.jsonl", "--name", key.upper()+"-val",
            "--out", f"work/validation/{key}.json")
        pred_file = f"work/{key}.jsonl"
        lab("infer", "--checkpoint", directory, "--split", "test", "--batch-size", INFER_BATCH, "--out", pred_file)
    lab("score", "--pred", pred_file, "--name", key.upper(), "--out", f"report/evidence/{key}.json")
    backup_reports()
lab("summarize")
backup_reports()
display(Markdown(Path("report/results.md").read_text()))
''')
md("""
### E5 bonus: reward IoU
`vlm_lab/reward.py` có reward Python thuần và test (hoàn hảo, JSON hỏng, sai lớp, thừa box, rỗng).
Phần GRPO chưa được triển khai/chạy; không ghi nhận bonus GRPO nếu chỉ chạy reward test.
""")
code('''
run_cmd(sys.executable, "-m", "pytest", "-q", "student_checks/test_reward.py")
''')
md("""
## 9. Báo cáo và xuất kết quả
Điền REPORT.md bằng số thật từ report/results.md, đọc ví dụ trong report/figures/*/index.md,
trả lời 10 câu bằng hiểu biết của bạn. Không thay template bằng câu trả lời giả để vượt checker.
Cell sau kiểm tra chấm lệch tọa độ cho Câu 3; kết quả nằm trong work, không thay evidence chuẩn.
""")
code('''
# Diagnostic only: chấm cùng output norm1000 dưới giả định pixel sai.
run_cmd(sys.executable, "scripts/score.py", "--pred", "work/e1.jsonl", "--coord", "pixel",
        "--name", "E1-wrong-scale-diagnostic", "--out", "work/e1_wrong_scale.json")
correct = json.loads(Path("report/evidence/e1.json").read_text())["micro"]["f1"]
wrong = json.loads(Path("work/e1_wrong_scale.json").read_text())["micro"]["f1"]
print("F1 correct scale:", correct, "| F1 wrong scale:", wrong)
''')
code('''
backup_reports()
# Chỉ đóng gói report/: không kèm data, secrets hoặc trọng số.
bundle = shutil.make_archive(str(ARTIFACT_DIR / "lab_results"), "zip", root_dir=REPO_DIR, base_dir="report")
print("Results ZIP:", bundle)
# FAIL lúc này có thể là REPORT còn trống hoặc repo chưa đủ commit. Xem từng mục, sửa thật rồi kiểm tra lại.
result = run_cmd(sys.executable, "scripts/check_submission.py", check=False)
print("Submission checker exit code:", result.returncode)
''')
md("""
Tải `lab_results.zip` từ Drive, chép `report/` về repo local. Điền REPORT.md, commit theo các bước
(ít nhất 5 commit có ý nghĩa), push repo công khai, chạy lại checker trên repo local.
Để nộp notebook có output: File → Download → Download .ipynb và thay bản trong repo.
Không push checkpoint/adapter/projector hoặc predictions trong work/. Không cần chạy GRPO để hoàn thành phần bắt buộc.
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"name": "python3", "display_name": "Python 3"},
      "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
for i, cell in enumerate(cells):
    cell["id"] = f"day10-{i:03d}"
Path(__file__).with_name("lab_day10.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
print(f"Wrote {len(cells)} cells")

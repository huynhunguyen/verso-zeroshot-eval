# VERSO — Gói Đánh Giá Zero-Shot Cho Mô Hình Thương Mại

Repo gọn nhẹ được tách riêng từ dự án **VERSO** để chạy thực nghiệm baseline **MLLM Zero-shot** (thí nghiệm `E-20`) trên các API thương mại (**Google Gemini** và **OpenAI**), với **cùng 100% tập đề, ảnh, prompt và bộ chấm tất định** như hệ thống chính.

## 1. Trong repo này có gì?

- **Dữ liệu đánh giá VN (`datasets/`):**
  - `1.765` câu hỏi chuẩn trong [circuits_full.json](datasets/circuit-dataset/circuits_full.json) (gồm `1.710` câu định lượng chấm tự động + `55` câu đáp án bằng lời để chờ chấm tay theo quy tắc D29).
  - Đúng `245` ảnh sơ đồ mạch trong `datasets/circuit-dataset/images/` (tổng ~3,4 MB).
- **Prompt chuẩn cố định (`configs/prompts/`):**
  - Mặc định dùng **`mllm_zero_shot_v1.txt`** (1 lượt gọi duy nhất: ảnh + đề → giải từng bước → chốt JSON cuối câu). **Không sửa nội dung file prompt** để giữ mã hash khớp tuyệt đối với repo chính.
- **Cấu hình model sẵn (`configs/models/`):**
  - **Google Gemini** (qua OpenAI-compatible endpoint của Google AI Studio):
    - `gemini-3-pro.yaml`
    - `gemini-3-flash.yaml`
  - **OpenAI** (qua `https://api.openai.com/v1`):
    - `openai-gpt-4o.yaml`
    - `openai-gpt-5.yaml`
    - `openai-o3.yaml`
- **Cơ chế an toàn khi chạy API dài:**
  - Tự động retry theo *exponential backoff* khi gặp lỗi `429 RateLimit` / `5xx` / timeout.
  - Tự động lưu **cache SQLite** (`.cache/llm.sqlite`) và **checkpoint từng câu** (`runs/<run_id>/checkpoint_questions.jsonl`) — lỡ mất mạng hoặc tắt máy có thể chạy tiếp bằng `--resume <RUN_ID>` mà **không tốn tiền gọi lại các câu đã xong**.

---

## 2. Cài đặt nhanh (Python >= 3.11)

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Hoặc Linux / macOS:
# source .venv/bin/activate

pip install -e ".[dev]"
pytest
```

Nếu `pytest` báo **4 passed** là môi trường và toàn bộ dữ liệu đã sẵn sàng.

---

## 3. Cài đặt API Key

Copy file `.env.example` thành `.env` rồi điền key tương ứng (file `.env` đã nằm trong `.gitignore`, **tuyệt đối không commit lên Git**):

```bash
cp .env.example .env
```

Nội dung `.env`:
```env
GEMINI_API_KEY=...
OPENAI_API_KEY=...
```

---

## 4. Quy trình chạy thí nghiệm (2 bước)

> **Quan trọng:** Luôn chạy thử **Bước 1 (`--limit 5`)** trước để kiểm tra API key, đúng tên model snapshot, định dạng JSON trả về và lượng token tiêu thụ, sau đó mới chạy **Bước 2 (full 1.765 câu)**.

### Bước 1 — Chạy thử 5 câu (Smoke Test)

```bash
# Google Gemini Pro (kiểm tra tên snapshot mới nhất trên Google AI Studio; có thể ghi đè bằng --model)
python -m verso_eval.baselines.mllm_zero_shot run --profile gemini-3-pro --limit 5

# Google Gemini Flash
python -m verso_eval.baselines.mllm_zero_shot run --profile gemini-3-flash --limit 5

# OpenAI GPT-5 / o3 / GPT-4o
python -m verso_eval.baselines.mllm_zero_shot run --profile openai-gpt-5 --limit 5
python -m verso_eval.baselines.mllm_zero_shot run --profile openai-o3 --limit 5
python -m verso_eval.baselines.mllm_zero_shot run --profile openai-gpt-4o --limit 5
```

> **Lưu ý nếu API báo model không tồn tại (`404 NotFound`):** Kiểm tra chính xác Model ID đang mở trên tài khoản của bạn rồi truyền trực tiếp qua cờ `--model <model_id>` (ví dụ `--model gemini-3-pro` hoặc `--model gpt-5-2025-08-07`) hoặc sửa trường `served_name:` trong file YAML tương ứng rồi commit lại trước khi chạy full.

### Bước 2 — Chạy chính thức toàn bộ 1.765 câu

*(Yêu cầu `git status` sạch và có đủ `--note`, `--server-info` để ghi dấu vết tái lập vào `meta.json` và `runs/registry.csv`.)*

```bash
# Ví dụ chạy Google Gemini Pro full 1.765 câu:
python -m verso_eval.baselines.mllm_zero_shot run \
  --profile gemini-3-pro \
  --workers 4 \
  --note "E-20 commercial zero-shot baseline: Gemini 3 Pro" \
  --server-info "Google AI Studio OpenAI-compat API; model gemini-3.1-pro-preview"

# Ví dụ chạy OpenAI GPT-5 full 1.765 câu:
python -m verso_eval.baselines.mllm_zero_shot run \
  --profile openai-gpt-5 \
  --workers 4 \
  --note "E-20 commercial zero-shot baseline: OpenAI GPT-5" \
  --server-info "OpenAI API; model gpt-5; reasoning_effort=medium"
```

*(Gợi ý chỉnh `--workers`: để `4` nếu tài khoản tier thấp để tránh 429; tier cao có thể tăng lên `8–12`.)*

### Nếu bị ngắt giữa chừng — Chạy tiếp (Resume)

Giữ **nguyên câu lệnh cũ** và thêm `--resume <RUN_ID>` (lấy tên thư mục trong `runs/`):

```bash
python -m verso_eval.baselines.mllm_zero_shot run \
  --profile gemini-3-pro \
  --workers 4 \
  --note "E-20 commercial zero-shot baseline: Gemini 3 Pro" \
  --server-info "Google AI Studio OpenAI-compat API; model gemini-3.1-pro-preview" \
  --resume 20261124-163000-mllm-zero-shot-gemini-3-pro-eval
```

---

## 5. Xem kết quả & Bàn giao lại cho nhóm

Sau khi chạy xong (hoặc đang chạy dở), chạy lệnh sau để xem độ chính xác định lượng (trên `1.710` câu chấm tự động), từng chủ đề (`DLO`, `RLC`, `TD`, `DDT`, `CH`), lỗi gọi API và tổng token:

```bash
python scripts/check_accuracy.py --latest
# Hoặc chỉ định rõ run_id:
python scripts/check_accuracy.py <RUN_ID>
```

**Dữ liệu cần gửi lại sau khi chạy xong:**
- Toàn bộ thư mục `runs/<RUN_ID>/` (gồm `config.json`, `meta.json`, `summary.json`, `trace.jsonl`) và file `runs/registry.csv`.
- **Không gửi** file `.env` và không cần gửi `.cache/llm.sqlite`.

# Hướng dẫn chạy thử code chính thức trên Google Colab

> Nhánh: `explore/official-code` (tách từ `main`). Mục đích: **chạy code của tác giả** (`third_party/vlm2rec_anon/`) để biết nó làm được gì, chạy ra số bao nhiêu so với paper, và còn thiếu gì. Đây **chưa phải** bản tái hiện của chúng ta (bản đó nằm ở `src/vlm2rec/` theo `PLAN.md`).
> Sau khi chạy xong, dùng `docs/official_code/AUDIT.md` để đánh giá.

## 0. Tổng quan

```
Notebook 01 (CPU, miễn phí)            Notebook 02 (GPU A100/L4)
┌─────────────────────────┐            ┌────────────────────────────────────────────┐
│ tải raw Amazon + meta   │            │ 0  setup + đúng phiên bản thư viện           │
│ tiền xử lý CHÍNH THỨC   │ ── Drive ─▶│ A  probe hành vi thực tế của code           │
│ tải ảnh                 │  (tar.gz)  │ B  SASRec chỉ ID (Stage 2 chính thức)        │
│ thống kê vs paper       │            │ C  VLM_Vanilla: full ranking + plug          │
└─────────────────────────┘            │ D  smoke VLM2Rec (256 user, 1 epoch)         │
                                       │ E  (tuỳ chọn) train đầy đủ 3 epoch          │
                                       │ F  bảng so sánh với paper                    │
                                       └────────────────────────────────────────────┘
```

Code gốc được **chép sang `/content/work/official`** trên Colab rồi mới chạy. Bản trong repo không bị sửa. Các script trong `tools/official/` chỉ *gọi* hàm của tác giả, và bổ sung 2 thứ code gốc không có: **đánh giá full ranking** (paper báo full ranking, còn code gốc chỉ so với 100 negative ngẫu nhiên) và **probe** kiểm tra hành vi.

| Script | Vai trò |
|---|---|
| `tools/official/prep_amazon.py` | Gọi tiền xử lý chính thức (song song phần tải mạng) + thống kê "Checkpoint A" |
| `tools/official/audit_probes.py` | Probe P1–P8: phiên bản, prompt, số token ảnh, WPCL β/τ, CRTR τ, giao thức eval, mặc định Stage 2 |
| `tools/official/fullrank_eval.py` | Đánh giá Stage 1 bằng full ranking + dropout test TEXT/VISION/FUSION + geometry; chạy được cả VLM_Vanilla |
| `tools/official/run_stage2.py` | Chạy `cf_baselines/main.py` (SASRec) với tham số mặc định của tác giả; mode `id_only / modality_only / concat` |
| `tools/official/summarize.py` | Gom JSON → bảng Markdown so với paper |

## 1. Chuẩn bị (làm một lần)

1. **Đưa nhánh lên GitHub:** `git push -u origin explore/official-code`
2. **Nếu repo private**, Colab cần token để clone:
   - GitHub → Settings → Developer settings → *Fine-grained tokens* → tạo token, chỉ cấp quyền repo `VLM2Rec`, *Contents: Read-only*.
   - Colab → biểu tượng 🔑 (*Secrets*) ở thanh trái → *Add new secret*: tên `GITHUB_TOKEN`, giá trị là token → bật *Notebook access*.
   - Repo public thì bỏ qua bước này.
3. **Google Drive** còn trống ≥ 3 GB. Mọi thứ lưu ở `MyDrive/VLM2Rec_work/`: `data/`, `audit_results/`, `checkpoints/`, `results/`, `logs/`.
4. **Colab:** notebook 01 chạy CPU (miễn phí). Notebook 02 cần GPU **A100** (khuyến nghị) hoặc **L4**, tức cần Colab Pro/Pro+. **T4 không dùng được** vì không có bf16.

## 2. Mở notebook

- **Repo public:** mở link
  - `https://colab.research.google.com/github/TrNguyenMQuan/VLM2Rec/blob/explore/official-code/notebooks/official/01_official_data_prep_cpu.ipynb`
  - `https://colab.research.google.com/github/TrNguyenMQuan/VLM2Rec/blob/explore/official-code/notebooks/official/02_official_stage1_stage2_gpu.ipynb`
- **Repo private:** Colab → *File → Open notebook → GitHub* → tick *Include private repos* → chọn repo, nhánh `explore/official-code`, rồi chọn notebook.
- Hoặc tải file `.ipynb` trong `notebooks/official/` lên qua *File → Upload notebook*.

> Muốn lưu thay đổi của notebook: *File → Save a copy in Drive*. Notebook trong repo nên giữ nguyên làm bản gốc.

## 3. Notebook 01 — chuẩn bị dữ liệu (CPU)

| Bước | Làm gì | Kỳ vọng |
|---|---|---|
| Cấu hình | `DATASET = "Beauty"` (nên bắt đầu với Beauty: paper dùng nó cho mọi phân tích) | — |
| Mount + clone | cho phép truy cập Drive | in ra commit mới nhất của nhánh |
| Cài thư viện | `pandas<3` (code gốc hỏng với pandas 3) | in phiên bản |
| Tiền xử lý | `prep_amazon.py` (~15–30 phút) | in thống kê và tiến trình tải ảnh |
| Bảng so sánh | in `paper table` / `checkpoint_A` / `official_output` | **Checkpoint A ≈ 22,363 / 12,077 / 198,064** (Beauty) |
| Lưu Drive | `data/Beauty_processed.tar.gz` | file vài trăm MB |

Cần ghi lại: Checkpoint A có khớp paper không; `official_output` (sau khi cắt 10 item) còn bao nhiêu user/item; số ảnh lỗi.

## 4. Notebook 02 — chạy Stage 1 + Stage 2 (GPU)

Chạy từ trên xuống. Thời gian ghi dưới đây là **ước tính** (chưa đo), và chính notebook sẽ đo thực tế.

| Phần | Làm gì | Cần nhìn vào đâu | Paper (Beauty) |
|---|---|---|---|
| 0 Setup | kiểm GPU/bf16, clone, giải nén dữ liệu, cài `transformers==4.55.0 peft==0.17.0 qwen-vl-utils==0.0.14 accelerate` | không có lỗi; phiên bản đúng | — |
| A Probe | `audit_probes.py` | các dòng `[PROBE ..] OK/DIFF/WARN` | — |
| B SASRec ID | `run_stage2.py --mode id_only` | H@10/H@20/N@10/N@20 | .0713 / .1053 / .0397 / **.0483** |
| C Vanilla | `fullrank_eval.py` (không adapter) rồi plug với 2 mode | `main` (FUSION, masked, dot), `dropout`, `sampled100`; plug: `modality_only` vs `concat` | Stage 1 **.0442**; plug **.0463** |
| D Smoke | `main.py` của tác giả: `--data_sample_K 256 --epochs 1 --phase embed` | có chạy hết không, loss có giảm không, thời gian/bước, VRAM | (không so số) |
| E Full (tuỳ chọn) | `RUN_FULL=True`: 3 epoch, batch 8 | Stage 1 & plug | Stage 1 **.0784**; plug **.0728** |
| F Tổng hợp | `summarize.py` | bảng Markdown `summary_Beauty.md` | — |

Mẹo:
- Muốn tiết kiệm thời gian ở Phần C thì đặt `EVAL_MAX_USERS = 5000`. Số sẽ dao động hơn, nên bản cuối hãy dùng `0` (toàn bộ user).
- Muốn có trung bình 5 seed như paper thì đặt `STAGE2_SEEDS = "42 43 44 45 46"`. Stage 2 chạy nhanh, nên rất đáng làm cho Phần B và E.
- Phần E: trainer của tác giả **không resume được**. Nếu runtime ngắt giữa chừng, phải chạy lại Phần 0 rồi Phần E. Checkpoint epoch 1 vẫn còn trên Drive và có thể đem đi đánh giá.

## 5. Đưa kết quả về repo và đánh giá

1. Trên Drive, tải `MyDrive/VLM2Rec_work/audit_results/` gồm `prep_report_*.json`, `probes_*.json`, `fullrank_*.json`, `stage2_*.json`, `summary_*.md`. **Không** tải checkpoint.
2. Chép vào `docs/official_code/results/` trên nhánh này → commit.
3. Mở `docs/official_code/AUDIT.md` và điền mục 3–5, hoặc nói với Claude: *"đọc docs/official_code/results và cập nhật AUDIT.md + DECISIONS.md"*.

## 6. Lỗi thường gặp

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `process_vision_info() got an unexpected keyword argument 'image_patch_size'` | qwen-vl-utils cũ | `pip install qwen-vl-utils==0.0.14` |
| `ValueError: ... is already used by a Transformers model` | sai bản transformers (`models.py` đăng ký lại lớp Qwen) | `pip install transformers==4.55.0` |
| `... requires accelerate` | `device_map="auto"` cần accelerate (requirements của tác giả thiếu) | `pip install accelerate` |
| `KeyError: 'user_id'` khi tiền xử lý | pandas 3 đổi hành vi của `groupby().apply` | `pip install "pandas<3"` |
| `FileNotFoundError: .../<sid>.jpg` | thiếu ảnh | chạy lại notebook 01 (ảnh đã có sẽ được bỏ qua) |
| HTTP 403/429 khi tải ảnh | tải quá nhanh | giảm `WORKERS` xuống 4–8 rồi chạy lại |
| `CUDA out of memory` | GPU nhỏ / batch lớn | dùng A100; hoặc giảm `FULL_BATCH` (ghi lại vào AUDIT) |
| GPU là T4 / `bf16` assert | T4 không có bf16 | đổi runtime sang A100/L4 |
| Lỗi lạ ở torch mới của Colab | Colab dùng torch mới hơn bản 2.7 của tác giả | chạy cell tuỳ chọn cài `torch==2.7.0` → *Runtime → Restart* → chạy lại từ Phần 0 |
| Clone thất bại (repo private) | thiếu secret `GITHUB_TOKEN` | xem mục 1.2 |

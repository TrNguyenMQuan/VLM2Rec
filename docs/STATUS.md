# STATUS — trạng thái hiện tại

**Cập nhật lần cuối:** 2026-11-25 (session 2)
**Nhánh hiện tại:** `explore/official-code` (tách từ `main`) — chạy thử và đánh giá code chính thức trên Colab.
**Phase:** 0.5 — Kiểm tra code chính thức (trước Phase 0 của bản tái hiện).

## Đã xong
- Session 1: đọc paper, lấy code chính thức, lập `PLAN.md`, `docs/*` (xem SESSION_LOG).
- Session 2: tạo nhánh `explore/official-code` với:
  - `notebooks/official/01_official_data_prep_cpu.ipynb` (CPU) và `02_official_stage1_stage2_gpu.ipynb` (A100/L4).
  - `tools/official/`: `prep_amazon.py`, `audit_probes.py`, `fullrank_eval.py`, `run_stage2.py`, `summarize.py`, `paper_targets.py`.
  - `docs/official_code/GUIDE_COLAB.md` (hướng dẫn chạy) và `AUDIT.md` (khung đánh giá 3 lớp + bảng cần điền).
  - Đã kiểm tra cục bộ: biên dịch OK; selftest xếp hạng khớp brute-force; parser tham số và log của Stage 2 OK; summarize OK trên dữ liệu giả. **Chưa chạy gì trên GPU.**
  - Đã xác minh: cần `qwen-vl-utils==0.0.14` (bản 0.0.11 thiếu `image_patch_size`); ảnh 112×112 → 16 token.

## Việc tiếp theo (theo thứ tự)
1. Bạn: `git push -u origin explore/official-code` (nếu repo private thì thêm secret `GITHUB_TOKEN` trên Colab).
2. Bạn: chạy notebook 01 (CPU, Beauty) → notebook 02 Phần 0, A, B, C, D (A100/L4). Phần E (train đầy đủ) tuỳ ngân sách.
3. Bạn: tải `audit_results/` từ Drive → `docs/official_code/results/` → commit.
4. Claude: đọc kết quả → điền `AUDIT.md` mục 3–5 → cập nhật `DECISIONS.md` (D-01, D-08, D-09, có thể cả D-03/04/05) và `RESULTS.md`.
5. Sau đó quay về `main` để bắt đầu Phase 0 của bản tái hiện (`PLAN.md`).

## Đang chờ từ bạn
- Kết quả chạy Colab (mục 2–3 ở trên).
- (Vẫn mở từ session 1) TMDB API key; lựa chọn GPU và ngân sách; phạm vi (MVP hay đủ 5 dataset).

## Quyết định sẽ được kiểm chứng bởi lần chạy này
- D-01 chế độ dữ liệu (SASRec ID trên tail10 có khớp paper không).
- D-08 giao thức eval Stage 1 (full ranking masked/unmasked vs sampled-100).
- D-09 cách cắm bảng item ở plug (`modality_only` vs `concat`).
- D-03/D-04/D-05 nếu chạy Phần E (train đầy đủ).

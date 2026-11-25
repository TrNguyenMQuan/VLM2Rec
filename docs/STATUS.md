# STATUS — trạng thái hiện tại

**Cập nhật lần cuối:** 2026-11-25 (session 1)
**Phase hiện tại:** Phase 0 — chưa bắt đầu viết code. Đã xong khâu đọc paper và lập kế hoạch.

## Đã xong
- Đọc toàn bộ paper v3 (TeX trong `paper/arxiv_2603.17450v3/`), tóm tắt công thức + mọi số mục tiêu vào `docs/PAPER_NOTES.md`.
- Tìm và tải code chính thức (ẩn danh) → `third_party/vlm2rec_anon/`; phân tích khác biệt paper ↔ code vào `docs/REFERENCE_CODE_NOTES.md` (C1–C12).
- Lập `PLAN.md`, `docs/DECISIONS.md` (D-01…D-22), `CLAUDE.md`, lệnh `/session-end`, `/log-run`.
- Kiểm tra nguồn: Amazon 2014 5-core + meta theo ngành online; 25/25 URL ảnh mẫu OK sau khi đổi sang `m.media-amazon.com`; ML-1M online; Qwen2.5-VL-3B-Instruct không gated; tokenizer mặc định right padding.

## Việc tiếp theo (theo thứ tự)
1. Chờ bạn trả lời các câu hỏi ở mục "Đang chờ".
2. Phase 0: `git init` (nếu được đồng ý) → `uv` project Python 3.11 với các bản khoá → khung `src/vlm2rec/utils` → tải Qwen2.5-VL-3B-Instruct → smoke test MPS.
3. Phase 1.1: tải raw Amazon (4 ngành, 5-core + meta) và ML-1M + links.csv.

## Đang chờ từ bạn
- TMDB API key (poster ML-1M) — D-14.
- Chọn GPU cho Stage 1 (Colab Pro/Pro+ hay thuê RunPod/Vast) + ngân sách.
- Đồng ý `git init` / tạo repo GitHub private?
- Phạm vi: MVP (ML-1M + Beauty) hay đủ 5 dataset + Tier 2.

## Quyết định đang mở quan trọng nhất
- D-01 chế độ dữ liệu (`tail10` tạm chọn) → kiểm chứng ở Phase 2 bằng SASRec ID-only.
- D-04/D-05 variant loss (`code` tạm chọn) → kiểm chứng ở Phase 4 trên Beauty/ML.
- D-09 bảng item plug (`modality_only` tạm chọn) → kiểm chứng bằng VLM_Vanilla plug.

# SESSION_LOG — nhật ký các session (chỉ ghi thêm, mới nhất ở cuối)

Mẫu cho mỗi mục:
```
## Session N — YYYY-MM-DD
- Mục tiêu:
- Đã làm:
- Phát hiện / bất ngờ:
- Run đã tạo: (run_id → kết quả chính)
- Quyết định thay đổi: (D-xx: ...)
- Dở dang / việc tiếp theo:
```

## Session 1 — 2026-11-25
- **Mục tiêu:** đọc paper arXiv:2603.17450, lập kế hoạch tái hiện và hệ thống file ghi nhớ.
- **Đã làm:**
  - Tải TeX source v3 của paper (ở trang abs không có link GitHub). Đọc mọi section, appendix, bảng và hình (một số số liệu đọc từ biểu đồ, sai số ±0.001).
  - Phát hiện link code ẩn danh bị comment trong abstract → tải qua API `anonymous.4open.science/api/repo/VLM2Rec/file/<path>` (đường dẫn web `/r/` trả 401). Đã có đủ 25 file gồm cả `datasets/` và `cf_baselines/`.
  - Viết `PLAN.md`, `CLAUDE.md`, `docs/*`, `.claude/commands/{session-end,log-run}.md`, `.gitignore`.
- **Phát hiện:**
  - Code cắt chuỗi còn 10 item cuối và resize ảnh 112×112 (16 token/ảnh) → giải thích được vì sao train vừa trên RTX 3090.
  - Bảng thống kê của paper ứng với "5-core + join meta", không cắt chuỗi, không áp 5-core lại (users Beauty/Clothing giữ nguyên).
  - SASRec ML-1M trong paper (H@10 0.1263) thấp hơn nhiều so với SASRec trên ML-1M đầy đủ → nghiêng về việc tác giả dùng dữ liệu đã cắt.
  - Loss trong code khác công thức paper (C1–C5, C12); eval Stage 1 trong code là sampled-100 dù paper báo full ranking; Stage 2 mặc định ghép ID embedding.
  - Thời gian trong paper (~93 phút/epoch Beauty) khớp với 1 mẫu/user; trục x của hình gradient (~12k bước) chưa giải thích được.
- **Run đã tạo:** chưa có.
- **Quyết định:** tạo D-01…D-22 (phần lớn PROVISIONAL theo nguyên tắc P-1: theo code).
- **Việc tiếp theo:** xem `docs/STATUS.md`.

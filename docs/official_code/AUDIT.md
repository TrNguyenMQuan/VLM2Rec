# Đánh giá code chính thức so với paper

> Đối tượng: `third_party/vlm2rec_anon/` (bản phát hành ẩn danh của tác giả). Cách chạy: `docs/official_code/GUIDE_COLAB.md`.
> Kết quả thô: `docs/official_code/results/` (JSON + `summary_<DS>.md` do notebook sinh ra).
> Ký hiệu: ✅ có và khớp paper · ⚠️ có nhưng khác paper · ❌ không có · ☐ chưa xác nhận bằng runtime · ☑ đã xác nhận.

## 1. Cách đánh giá — 3 lớp

| Lớp | Câu hỏi | Công cụ | Trạng thái |
|---|---|---|---|
| L1 · Review tĩnh | Code có những thành phần nào, viết khác paper ở đâu? | Đọc code → `docs/REFERENCE_CODE_NOTES.md` | ✅ xong (session 1) |
| L2 · Probe runtime | Khi chạy thật, hành vi có đúng như review tĩnh không? | Notebook 02 Phần A (`audit_probes.py`) | ☐ |
| L3 · Số liệu | Chạy ra số có khớp paper không? Khớp với cấu hình nào? | Notebook 01 + 02 Phần B–E + `summarize.py` | ☐ |

Nguyên tắc: một kết luận ở mục 5 chỉ được viết khi có bằng chứng L2 hoặc L3 (tên file JSON trong `results/`).

## 2. Đối chiếu thành phần (paper ↔ code)

| # | Thành phần trong paper | Mục paper | Code | Ghi chú từ review tĩnh | Xác nhận bằng | Runtime |
|---|---|---|---|---|---|---|
| **A** | **Dữ liệu** | | | | | |
| A1 | Amazon 2014 5-core, bỏ item thiếu title/ảnh | §6.1, App. A | ⚠️ | Có, nhưng còn cắt **10 item cuối** + lọc URL ảnh + áp 5-core lại (C9) | NB01: `checkpoint_A` vs `official_output` | ☐ |
| A2 | MovieLens-1M + poster TMDB | App. A | ❌ | Script không chạy được với ML-1M; không có phần tải poster | — | — |
| A3 | Leave-one-out | §6.1 | ✅ | Stage 1: train `seq[:-2]→seq[-2]`, test `seq[:-1]→seq[-1]`; Stage 2: có valid | — | — |
| A4 | Kích thước ảnh | (không nói) | ℹ️ | Resize 112×112 → 16 token/ảnh | P3b | ☐ |
| **B** | **Encoder Stage 1** | | | | | |
| B1 | Qwen2.5-VL-3B + LoRA r16/α32/p0.2 | App. A | ✅ | LoRA trên q,k,v,o,gate,up,down của LM; vision đóng băng | log Phần D | ☐ |
| B2 | Prompt tách modality P_T / P_V | §4 | ⚠️ | Khác chữ ("title", "one word.,", "0). "); title không qua chat template; ảnh có system prompt | P2a, P2e, P3a, P3c | ☐ |
| B3 | Last-token + L2 norm; fused = e^T + e^V | §3–4 | ✅ | Right padding; fused không chuẩn hoá lại | P3d | ☐ |
| B4 | Bảng item dùng cùng prompt với item đích lúc train | Alg. 1 | ⚠️ | Lúc sinh bảng item dùng `0){title}` và `{sid})`+ảnh, khác với lúc train | P2c, P2d | ☐ |
| B5 | 1 mẫu/user; AdamW 1e-5; 3 epoch | App. A | ✅ | Không scheduler, không clipping, không resume | log Phần D/E | ☐ |
| B6 | Batch 32 | App. A | ⚠️ | Mặc định 8 (README cũng ghi 8) | — | — |
| B7 | Few-shot K_u | §6.9 | ✅ | `--data_sample_K` | Phần D | ☐ |
| **C** | **Hàm loss** | | | | | |
| C1 | Baseline SFT (Eq. 2) | §3 | ❌ | `--loss_type` chỉ có `wpcl` | — | — |
| C2 | WPCL: margin theo user = pos − mean(neg); w = 1 + softplus(Δ) | §5.1 | ⚠️ | NT-Xent đối xứng 2B; margin theo **cặp**, đơn vị logit (÷τ); w = 1 + σ(0.8)·softplus(σ(0.5)·Δ) | P4, P6 | ☐ |
| C3 | Stop-gradient trên margin mạnh | §5.1 | ✅ | `.detach()` | — | — |
| C4 | CRTR: softmax(s/τ), KL 2 chiều | §5.2 | ⚠️ | softmax(s)/τ → τ thành **hệ số nhân loss** | P5 | ☐ |
| C5 | L = WPCL + λ·CRTR | §5.3 | ✅ | `--lmd` | — | — |
| C6 | Các biến thể ablation | §6.4 | ⚠️ | Chỉ tắt được CRTR (`--aux_loss none`); không có cờ cho no-stopgrad / no-w_pen / w/o WPCL | — | — |
| C7 | Grid τ_WPCL, τ_CRTR, λ | App. A | ⚠️ | Có tham số CLI, không có script sweep | — | — |
| **D** | **Đánh giá** | | | | | |
| D1 | Full ranking ở VLM2Rec setting | §6.1 | ❌ | Mặc định 100 negative ngẫu nhiên; nhánh full ranking gán mọi target = 0 (lỗi) | P7; bù bằng `fullrank_eval.py` | ☐ |
| D2 | H@10/20, N@10/20 | §6.1 | ✅ | — | — | — |
| D3 | 5 seed + paired t-test | §6.1 | ❌ | Chỉ có `--seed` | — | — |
| D4 | Dropout test TEXT/VISION/FUSION | §4, Fig. 1 | ❌ | Bù bằng `fullrank_eval.py` (`dropout_N@20_masked_dot`) | Phần C/E | ☐ |
| D5 | Gradient dynamics | §4 (A2) | ❌ | — | — | — |
| D6 | Geometry U / S | §4 (A3) | ❌ | Bù bằng `fullrank_eval.py --geometry` | Phần C/E | ☐ |
| **E** | **Plug (Stage 2)** | | | | | |
| E1 | SASRec + adapter W (chỉ W·e_i), d = 128 | §5.4 | ⚠️ | Mặc định `emb_fusion=concat` → [ID ‖ W·e_i], hidden 256 | P8a; Phần C so 2 mode | ☐ |
| E2 | Adam 1e-3, bs 256, dropout 0.5, patience 10 (valid N@20) | App. A | ✅ | max_len 50; CE trên toàn bộ item; mask lịch sử bằng 0 | P8b | ☐ |
| E3 | GRU4Rec, FMLP, DiffuRec | Tab. 10 | ❌ | Registry chỉ có SASRec | P8c | — |
| E4 | Hybrid UniSRec / TedRec / M3SRec / HM4SR | §6.6 | ❌ | — | — | — |
| **F** | **Baseline & phân tích khác** | | | | | |
| F1 | VLM_Vanilla | Tab. 2 | ❌ | Chạy được qua `fullrank_eval.py` không có adapter | Phần C | ☐ |
| F2 | VLM_SFT, LLM_*, BERT, CLIP, SLIM(+), LLMEmb, LLM2Rec, I-LLMRec, MLLM-MSR, NoteLLM-2, VLM_Prompt | Tab. 2 | ❌ | README ghi rõ không phát hành | — | — |
| F3 | OGM-GE / AGM | §6.5 | ❌ | — | — | — |
| F4 | Cold-start / cross-domain | §6.8 | ❌ | — | — | — |
| F5 | Phân tích theo user, biến thể penalty | §6.7 | ❌ | — | — | — |
| F6 | VLM khác (InternVL3, LLaVA, 7B/32B), Qwen3-VL-Embedding | App. B | ⚠️ | Có `--model_name` nhưng chỉ đăng ký lớp Qwen2-VL / Qwen2.5-VL | — | — |
| **G** | **Kỹ thuật** | | | | | |
| G1 | requirements đủ để chạy | — | ⚠️ | Thiếu `accelerate`; không pin `qwen-vl-utils` (cần 0.0.14) và `pandas` (<3) | P1 | ☐ |
| G2 | Checkpoint / resume | — | ❌ | Chỉ lưu epoch 1 và epoch cuối | — | — |
| G3 | Tái lập dữ liệu | — | ⚠️ | Lọc ảnh phụ thuộc mạng → dữ liệu có thể khác theo thời điểm chạy | NB01 `image_url_check` | ☐ |

## 3. Số liệu (Beauty) — điền sau khi chạy
Nguồn số "Ours": `docs/official_code/results/summary_Beauty.md`. Δ% = (Ours − Paper) / Paper.

### 3.1 Dữ liệu
| Giai đoạn | Users | Items | Interactions | Avg len | Ghi chú |
|---|---|---|---|---|---|
| Paper (bảng thống kê) | 22,363 | 12,077 | 198,064 | 8.9 | |
| Checkpoint A (ours) | | | | | |
| Output chính thức (tail10) | | | | | ảnh lỗi: |

### 3.2 Stage 2 — SASRec chỉ ID (dữ liệu tail10)
| | H@10 | H@20 | N@10 | N@20 | Seeds |
|---|---|---|---|---|---|
| Paper | 0.0713 | 0.1053 | 0.0397 | 0.0483 | 5 |
| Ours | | | | | |

### 3.3 VLM_Vanilla — VLM2Rec setting (full ranking)
| Biến thể | H@10 | H@20 | N@10 | N@20 |
|---|---|---|---|---|
| Paper | 0.0674 | 0.1054 | 0.0351 | 0.0442 |
| Ours FUSION, masked, dot | | | | |
| Ours FUSION, unmasked, dot | | | | |
| Ours sampled-100 (kiểu code gốc) | | | | |
| Dropout N@20 TEXT / VISION / FUSION — paper ≈ .045 / .042 / .0442 | | | | |

### 3.4 VLM_Vanilla — plug (SASRec)
| Mode | H@10 | H@20 | N@10 | N@20 |
|---|---|---|---|---|
| Paper | 0.0685 | 0.1031 | 0.0374 | 0.0463 |
| Ours `modality_only` | | | | |
| Ours `concat` | | | | |

### 3.5 Smoke (K_u = 256, 1 epoch)
| Chạy trọn? | Thời gian/bước | VRAM đỉnh | Loss đầu → cuối | Lỗi gặp phải |
|---|---|---|---|---|
| | | | | |

### 3.6 Full VLM2Rec (3 epoch, batch = ___)
| | H@10 | H@20 | N@10 | N@20 | Dropout T / V / F (gap) |
|---|---|---|---|---|---|
| Paper — Stage 1 | 0.1161 | 0.1732 | 0.0641 | 0.0784 | .078 / .077 / .0784 (1.3%) |
| Ours — Stage 1 | | | | | |
| Paper — plug | 0.1080 | 0.1511 | 0.0618 | 0.0728 | — |
| Ours — plug `modality_only` | | | | | — |
| Ours — plug `concat` | | | | | — |

### 3.7 Chi phí
| | Ours | Paper |
|---|---|---|
| GPU | | RTX 3090 |
| Thời gian / epoch (full) | | ~93 phút (Beauty) |
| Sinh bảng item | | 113 s |
| VRAM đỉnh khi train | | — |

## 4. Đọc kết quả → kết luận

| Quan sát | Kết luận | Cập nhật |
|---|---|---|
| Checkpoint A khớp paper (±0.5%) | Bảng thống kê của paper ứng với dữ liệu **chưa cắt** | ghi vào D-01 |
| SASRec ID trên tail10 khớp paper (±5%) | Paper train/eval trên dữ liệu **tail10** | đóng D-01 = `tail10` |
| SASRec ID lệch xa | Paper dùng dữ liệu khác → chạy lại NB01 với `--max_seq_len 0` để thử chế độ `full` | D-01 vẫn OPEN |
| Vanilla full ranking ≈ .0442, còn sampled-100 cao hơn nhiều | Paper dùng full ranking, không dùng eval của code | đóng D-08 (masked hay unmasked: chọn cái gần hơn) |
| Vanilla plug `modality_only` < SASRec ID (giống paper .0463 < .0483), còn `concat` ≥ SASRec | Paper dùng adapter-only, không dùng mặc định của code | đóng D-09 = `modality_only` |
| Smoke chạy trọn, loss giảm | Code chạy được end-to-end với đúng phiên bản thư viện | — |
| Full VLM2Rec Stage 1 & plug trong ±5% | Code chính thức **tái hiện được** paper → bản tái hiện của ta bám theo code (P-1) | D-04, D-05 → `code` (CLOSED) |
| Full lệch > 5% | Thử theo thang debug `PLAN.md` §7: batch 32 (D-03), mask/cosine (D-06/D-08), plug mode (D-09) | ghi vào DECISIONS |
| Dropout: gap TEXT–VISION nhỏ (≈1–2%) và FUSION > TEXT | Tái hiện được hiện tượng sửa modality collapse | ghi vào RESULTS |

## 5. Kết luận
> Phần "dự kiến" dưới đây rút ra từ review tĩnh (L1). Sau khi chạy, đánh dấu ☑ / ✗ và ghi bằng chứng.

### 5.1 Code chính thức đã làm được (dự kiến)
- ☐ Pipeline tiền xử lý Amazon end-to-end (tải, 5-core, join meta, cắt chuỗi, lọc ảnh, tải ảnh).
- ☐ Encoder Qwen2.5-VL-3B + LoRA với WPCL + CRTR, lấy last-token, fused bằng tổng, few-shot `K_u`.
- ☐ Sinh bảng embedding item (text / ảnh / fused).
- ☐ VLM2Rec_plug với SASRec (full ranking, early stopping theo valid N@20).

### 5.2 Còn thiếu so với paper (dự kiến)
- ☐ MovieLens-1M (cả tiền xử lý lẫn poster).
- ☐ Đánh giá full ranking cho VLM2Rec setting (con số chính của paper).
- ☐ Mọi baseline (VLM_Vanilla, VLM_SFT, LLM_*, BERT, CLIP, …) và mọi ablation trừ "w/o CRTR".
- ☐ Backbone GRU4Rec / FMLP / DiffuRec; mô hình hybrid.
- ☐ Phân tích: dropout test, gradient dynamics, geometry, theo user, cold-start, cross-domain, OGM-GE/AGM.
- ☐ Chạy nhiều seed và t-test; resume khi train.

### 5.3 Khác paper — có thể ảnh hưởng tới số (dự kiến)
- ☐ Dữ liệu cắt 10 item cuối (paper không nói).
- ☐ Công thức WPCL và CRTR khác phương trình trong paper.
- ☐ Batch 8 thay vì 32.
- ☐ Plug mặc định ghép ID embedding.
- ☐ Prompt lúc sinh bảng item khác prompt lúc train.

### 5.4 Hệ quả cho bản tái hiện của chúng ta
- (điền sau khi có số: những quyết định D-xx nào được đóng, và theo hướng nào)

# PLAN — Tái hiện VLM2Rec (arXiv:2603.17450v3)

> **Mục tiêu:** dựng lại code của paper *VLM2Rec: Resolving Modality Collapse in Vision-Language Models for Sequential Recommendation* (WWW '27) và đạt kết quả gần hoặc bằng số công bố.
> **Ngày lập:** 2026-11-25 · **Trạng thái hiện tại:** xem `docs/STATUS.md`
> Tài liệu kèm theo: `docs/PAPER_NOTES.md` (công thức + toàn bộ số mục tiêu), `docs/REFERENCE_CODE_NOTES.md` (phân tích code chính thức), `docs/DECISIONS.md` (các điểm mơ hồ), `docs/RESULTS.md` (bảng so sánh).

---

## 1. Tóm tắt paper (đủ để bắt tay vào làm)

**Ý tưởng.** Dùng VLM (Qwen2.5-VL-3B) làm *bộ mã hoá* cho sequential recommendation: đưa cả chuỗi tiêu đề và cả chuỗi ảnh của lịch sử user vào VLM, lấy hidden state của token cuối làm embedding. Tác giả phát hiện **modality collapse**: VLM dựa quá nhiều vào text; contrastive SFT (cần thiết để đưa CF vào) còn làm việc này tệ hơn — gradient đi theo text, không gian ảnh gần như không phân biệt được item kế tiếp với negative, cộng embedding ảnh vào embedding text còn làm **giảm** độ chính xác.

**Phương pháp** (chỉ đổi objective, không đổi kiến trúc):
1. Mã hoá **riêng** từng modality bằng 2 prompt (`P_T` cho chuỗi title, `P_V` cho chuỗi ảnh); embedding = last-token hidden state đã chuẩn hoá L2; fused = `e^T + e^V`.
2. **WPCL** (Weak-modality Penalized Contrastive Learning): với mỗi user, đo margin (pos − mean neg) của từng modality → modality yếu/mạnh → `Δgap = sg[M_strong] − M_weak` → trọng số `w = 1 + softplus(Δgap)` nhân vào tổng negative của InfoNCE trên embedding fused.
3. **CRTR** (Cross-modal Relational Topology Regularization): KL hai chiều giữa phân phối softmax-similarity của user với các item trong batch ở không gian text và không gian ảnh.
4. `L = L_WPCL + λ·L_CRTR`, chỉ train LoRA (r16, α32, dropout 0.2), AdamW lr 1e-5, batch 32, 3 epoch.
5. **VLM2Rec_plug:** tính sẵn bảng embedding item (fused) → qua adapter tuyến tính W (→ d=128) → thay bảng item của SASRec (và GRU4Rec, FMLP, DiffuRec); chỉ train W + backbone.

**Đánh giá:** Amazon 2014 (Toys, Beauty, Clothing, Sports) + MovieLens-1M; 5-core, leave-one-out; full ranking; H@10/20, N@10/20; trung bình 5 seed.

**Số mục tiêu chính (N@20):**

| Setting | Method | Toys | Beauty | Clothing | Sports | ML-1M |
|---|---|---|---|---|---|---|
| — | SASRec (ID) | 0.0602 | 0.0483 | 0.0260 | 0.0294 | 0.0773 |
| VLM2Rec | VLM_Vanilla | 0.0570 | 0.0442 | 0.0245 | 0.0271 | 0.0711 |
| VLM2Rec | VLM_SFT | 0.0732 | 0.0695 | 0.0325 | 0.0358 | 0.1087 |
| VLM2Rec | **VLM2Rec** | **0.0823** | **0.0784** | **0.0370** | **0.0396** | **0.1252** |
| plug (SASRec) | VLM_Vanilla | 0.0577 | 0.0463 | 0.0240 | 0.0262 | 0.0695 |
| plug (SASRec) | VLM_SFT | 0.0699 | 0.0622 | 0.0325 | 0.0326 | 0.1082 |
| plug (SASRec) | **VLM2Rec_plug** | **0.0786** | **0.0728** | **0.0345** | **0.0378** | **0.1216** |

Bảng đầy đủ (H@10/20, N@10/20, mọi baseline) và các bảng phụ: `docs/PAPER_NOTES.md` §6–7.

---

## 2. Phát hiện quan trọng khi đọc paper và code

1. **Có code chính thức (ẩn danh).** Link bị comment trong abstract (`anonymous.4open.science/r/VLM2Rec/`) vẫn tải được qua API. Đã lưu bản sao chỉ-đọc tại `third_party/vlm2rec_anon/`. Bản này có: tiền xử lý Amazon, Stage 1 (Qwen2.5-VL + LoRA + WPCL + CRTR), Stage 2 (SASRec). **Không có:** tiền xử lý ML-1M/poster TMDB, baseline, ablation, các backbone khác, eval full-ranking cho Stage 1.
2. **Code khác paper ở nhiều điểm** (bảng C1–C12 trong `docs/REFERENCE_CODE_NOTES.md`). Những điểm ảnh hưởng lớn nhất:
   - Tiền xử lý **cắt mỗi user còn 10 item gần nhất**, lọc URL ảnh rồi áp 5-core lại → khác bảng thống kê trong paper (ML-1M avg len 165).
   - Ảnh resize **112×112** → 16 visual token/ảnh. Đây là lý do việc train vừa với RTX 3090.
   - WPCL trong code là NT-Xent đối xứng trên 2B mẫu, margin tính theo *cặp* trên logit đã chia τ, `w = 1 + σ(0.8)·softplus(σ(0.5)·Δ)`; CRTR chia τ **sau** softmax (τ thành hệ số nhân loss).
   - Eval Stage 1 trong code dùng 100 negative ngẫu nhiên, nhưng số trong paper là full ranking.
   - Stage 2 mặc định ghép `[ID embedding ‖ W·e_i]`, trong khi paper mô tả chỉ dùng `W·e_i`.
   - Batch size mặc định 8, paper ghi 32.
3. **Chiến lược** (nguyên tắc P-1 trong DECISIONS): mặc định **theo code** (vì số trong paper do code sinh ra), mọi điểm khác biệt là một flag `variant: code|paper`, và dùng các **phép thử chẩn đoán rẻ** để quyết định:
   - SASRec ID-only (chạy trên Mac, vài phút) trên 2 chế độ dữ liệu, so với hàng SASRec của paper → quyết định chế độ dữ liệu (D-01).
   - VLM_Vanilla (không cần train) → kiểm tra prompt, pooling, ảnh, cách đánh giá (D-07, D-08, D-09) trước khi tốn GPU để train.
4. **Nguồn dữ liệu còn sống** (đã kiểm tra 2026-11-25): file 5-core và `meta_<Category>.json.gz` của Amazon 2014 trên snap.stanford.edu; 25/25 URL ảnh mẫu tải được sau khi đổi `ecx.images-amazon.com` → `m.media-amazon.com`; ML-1M trên grouplens; Qwen2.5-VL-3B-Instruct không bị gated.

---

## 3. Môi trường & compute

### 3.1 Phân vai thiết bị
| Việc | Mac M4 Pro 24 GB (MPS) | Colab (L4 24 GB / A100 40 GB) | GPU thuê (RTX 3090/4090/A100) + Docker |
|---|---|---|---|
| Phát triển code, unit test | ✅ chính | — | — |
| Tải & tiền xử lý dữ liệu, ảnh | ✅ chính | — | — |
| Stage 2 (SASRec/GRU4Rec/FMLP) | ✅ chính (vài phút/run) | được | được |
| Smoke test Stage 1 (K_u=64, vài bước) | ✅ | — | — |
| VLM_Vanilla embedding (chỉ inference) | ✅ cho ML-1M; Amazon chạy qua đêm | ✅ | ✅ |
| Few-shot Stage 1 (K_u ≤ 1024) | ⚠️ được nhưng chậm (ước 1.5–3 h/run) | ✅ | ✅ |
| Train Stage 1 đầy đủ (SFT, VLM2Rec, ablation) | ❌ quá chậm | ✅ cho MVP (Beauty + ML) | ✅ **khuyến nghị** cho toàn bộ |

Lưu ý:
- **Docker trên macOS không truy cập được GPU/MPS** → container chỉ dùng cho máy CUDA (RunPod, Vast.ai, server lab). Trên Mac chạy trực tiếp bằng `uv`.
- **T4 (Colab free) không hỗ trợ bf16** → không dùng cho Stage 1.
- Colab hay ngắt phiên → trainer phải lưu checkpoint mỗi N bước và resume được; dữ liệu đã xử lý, adapter LoRA và embedding lưu trên Google Drive.
- Kết quả cuối cùng chỉ lấy từ CUDA; số trên MPS chỉ để debug (khác biệt số học).

### 3.2 Ước tính chi phí Stage 1 (sẽ đo lại thật ở Phase 3)
Dựa trên thời gian trong paper (RTX 3090: Toys ~75 phút/epoch, Beauty ~93 phút/epoch ≈ 4 user/s) và giả định 1 mẫu/user:

| Dataset | #user | 3 epoch trên 3090 (ước) |
|---|---|---|
| ML-1M | 6,040 | ~1.3 h |
| Toys | 19,406 | ~3.8 h |
| Beauty | 22,363 | ~4.7 h |
| Sports | 35,597 | ~7.3 h |
| Clothing | 39,387 | ~8 h |
| **Tổng / 1 phương pháp** | | **~25 h** |

- Tier 1 (VLM_SFT + VLM2Rec + kiểm chứng variant trên Beauty/ML) ≈ 55–60 GPU-giờ hạng 3090; A100 ước nhanh hơn 2–2.5 lần, 4090 nhanh hơn 1.6–2 lần.
- MVP (chỉ ML-1M + Beauty, SFT + VLM2Rec) ≈ 12 h trên 3090 / ~5 h trên A100 → vừa với Colab Pro nếu có checkpoint/resume.
- Encode chuỗi user để đánh giá + bảng item: thêm khoảng 10–30 phút mỗi dataset mỗi model.
- Giá thuê GPU và số compute unit của Colab thay đổi theo thời điểm; cần kiểm tra lại trước khi chọn.

### 3.3 Phần mềm (khóa phiên bản theo code tác giả)
- Python 3.11 (qua `uv`), `torch==2.7.0`, `transformers==4.55.0`, `peft==0.17.0`, `qwen-vl-utils` (bản có tham số `image_patch_size`), `numpy`, `pandas`, `scipy`, `Pillow`, `requests`, `tqdm`, `pyyaml`, `pytest`, `ruff`.
- Hai nhóm extra trong `pyproject.toml`: `mac` (torch có MPS) và `cuda` (torch cu126). Biến môi trường trên Mac: `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Thứ tự thiết bị: `cuda > mps > cpu` (helper `utils/device.py`).

---

## 4. Cấu trúc repo dự kiến

```
VLM2Rec/
├── CLAUDE.md                 # hướng dẫn cho Claude (tự nạp mỗi session, import docs/STATUS.md)
├── PLAN.md                   # file này
├── pyproject.toml, uv.lock
├── configs/
│   ├── data/    {toys,beauty,sports,clothing,ml1m}.yaml
│   ├── stage1/  {vanilla,sft,vlm2rec,ablation_*,llm_sft}.yaml
│   └── stage2/  {sasrec,gru4rec,fmlp}.yaml
├── src/vlm2rec/
│   ├── utils/    config.py, rundir.py, seed.py, device.py, io.py
│   ├── data/     download.py, preprocess.py, images.py, movielens.py, stats.py
│   ├── stage1/   prompts.py, collate.py, embedder.py, losses.py, trainer.py, export.py, evaluate.py
│   ├── stage2/   dataset.py, item_table.py, models/{sasrec,gru4rec,fmlp}.py, trainer.py
│   ├── eval/     metrics.py, ranking.py, significance.py
│   └── analysis/ dropout_test.py, geometry.py, grad_dynamics.py, user_gap.py, cold_start.py
├── scripts/      prepare_data.sh, stage1.sh, stage2.sh, seeds.sh, aggregate.py
├── tests/        test_metrics.py, test_losses.py, test_prompts.py, test_preprocess.py, test_stage2.py
├── notebooks/    colab_stage1.ipynb
├── docker/       Dockerfile.cuda
├── docs/         STATUS.md, SESSION_LOG.md, DECISIONS.md, RESULTS.md, PAPER_NOTES.md, REFERENCE_CODE_NOTES.md
├── paper/arxiv_2603.17450v3/   # TeX + hình gốc (chỉ đọc)
├── third_party/vlm2rec_anon/   # code chính thức (chỉ đọc, không import)
├── data/         # gitignored: raw/, processed/<ds>/<regime>/
├── artifacts/    # gitignored: hf_cache/, adapters/, item_tables/
└── runs/         # <run_id>/config.yaml, metrics.json, env.json, log.txt (checkpoint/embedding bị gitignore)
```

**Quy ước run:** `run_id = YYYYMMDD-HHMM_<stage>_<dataset>_<method>[_<tag>]_s<seed>`. Mỗi run có `config.yaml` (đã resolve đầy đủ), `metrics.json`, `env.json` (git sha, phiên bản thư viện, thiết bị), `log.txt`. Không có run dir thì không được ghi số vào `docs/RESULTS.md`.

**Định dạng dữ liệu (giữ tương thích với code gốc):** `<ds>.txt` (`uid sid1 sid2 ...`, id bắt đầu từ 1), `<ds>_item_meta.csv` (`sid,asin|movie_id,title,imgurl`), `images/<sid>.jpg` (bản gốc) và `images_112/<sid>.png` (bản đã resize để tăng tốc).

---

## 5. Các phase

Mỗi phase có **Gate** (tiêu chí qua). Chưa qua gate thì không sang phase sau, trừ khi đã ghi rõ lý do trong `docs/DECISIONS.md`.

### Phase 0.5 — Chạy thử & đánh giá code chính thức (Colab, nhánh `explore/official-code`)
- [ ] Notebook 01 (CPU): tiền xử lý chính thức cho Beauty + thống kê Checkpoint A.
- [ ] Notebook 02 (GPU): probe runtime, SASRec ID, VLM_Vanilla (full ranking + plug), smoke VLM2Rec; tuỳ chọn train đầy đủ.
- [ ] Điền `docs/official_code/AUDIT.md`; cập nhật D-01 / D-08 / D-09 (và D-03/04/05 nếu đã train đầy đủ).

Hướng dẫn: `docs/official_code/GUIDE_COLAB.md`. **Gate 0.5:** có `summary_Beauty.md` trong `docs/official_code/results/`, và AUDIT mục 5 đã được xác nhận bằng runtime.

### Phase 0 — Khởi tạo (Mac, ~0.5 ngày)
- [ ] `git init` + `.gitignore` (đã có) + commit đầu tiên (cần bạn đồng ý).
- [ ] `uv` project, Python 3.11, khoá phiên bản như §3.3; `ruff`, `pytest`.
- [ ] Khung `src/vlm2rec/utils`: config YAML + override qua CLI (`key=value`), tạo run dir, seed, chọn thiết bị, ghi `env.json`.
- [ ] Tải `Qwen/Qwen2.5-VL-3B-Instruct` vào `artifacts/hf_cache` (~7.5 GB).
- [ ] Smoke test trên MPS (bf16): encode 1 prompt text và 1 prompt ảnh 112×112 → vector 2048 chiều, chuẩn L2; in số visual token (kỳ vọng 16), đúng vị trí token cuối với right padding.

**Gate 0:** `uv run pytest` xanh; smoke test in đúng kích thước/số token; `docs/STATUS.md` đã cập nhật.

### Phase 1 — Dữ liệu (Mac, ~1–2 ngày, phần lớn thời gian là chờ tải)
- [ ] **1.1 Tải raw:** `reviews_{Beauty,Toys_and_Games,Sports_and_Outdoors,Clothing_Shoes_and_Jewelry}_5.json.gz` + `meta_<Category>.json.gz` (D-15); `ml-1m.zip`; `links.csv` từ ml-25m hoặc ml-latest để phủ đủ movieId.
- [ ] **1.2 Tiền xử lý** (port theo đúng thứ tự của `preprocess_data.py`, có flag): rating > 0 → 5-core lặp → sort ổn định (user, time) → inner join meta (title & ảnh khác null) → **[Checkpoint A: so bảng thống kê paper]** → nếu `regime=tail10`: tail(10) → lọc ảnh tải được → 5-core lại → đánh id theo thứ tự xuất hiện → ghi file; làm sạch title (`,` `;` → khoảng trắng). Xuất cả `full` và `tail10`.
- [ ] **1.3 Ảnh Amazon:** đổi URL sang `m.media-amazon.com`, tải song song (≤ 8 luồng, có retry/backoff), lưu bản gốc và bản 112×112 bicubic; ghi danh sách lỗi. Tải một lần và dùng luôn kết quả làm bước "lọc ảnh hợp lệ".
- [ ] **1.4 ML-1M:** title từ `movies.dat` (latin-1); `movieId → tmdbId` (dự phòng `imdbId` qua `/find`) → `poster_path` → `image.tmdb.org/t/p/w342`. **Cần TMDB API key của bạn** (D-14). Kỳ vọng ≈ 3,397 phim có poster.
- [ ] **1.5 Báo cáo:** script thống kê (users, items, interactions, avg len, sparsity) + hash nội dung file → ghi vào `docs/RESULTS.md §Dữ liệu`.
- [ ] Unit test: 5-core, leave-one-out, ánh xạ id, làm sạch title.

**Gate 1:** Checkpoint A lệch ≤ 0.5% so với bảng thống kê của paper ở cả 5 dataset (Amazon kỳ vọng khớp chính xác); 100% item giữ lại đều có ảnh; chạy lại cho ra cùng hash.

### Phase 2 — Stage 2 + bộ đánh giá (Mac, ~1–2 ngày)
- [ ] Port SASRec giữ đúng ngữ nghĩa code (2 layer, 2 head, inner 256, GELU, dropout 0.5, max_len 50, CE trên toàn bộ item, Adam 1e-3, batch 256, ≤300 epoch, patience 10 theo valid N@20).
- [ ] Dataset: train = mọi prefix của `seq[-(max_len+2):-2]`; valid = `seq[:-1]`; test = toàn chuỗi.
- [ ] Full ranking có mask lịch sử (gán 0 như code, tuỳ chọn −inf — D-10); metric H/N@{5,10,20}; unit test với ranking đồ chơi.
- [ ] Các mode bảng item: `id_only | modality_only | concat | add` (D-09); loader kiểm tra **thứ tự sid** (code gốc dựa vào tên file — dễ lệch). Mode dropout test cho plug: history dùng e^T / e^V / fused, item đích luôn fused.
- [ ] Runner nhiều seed (42–46) + tổng hợp mean ± std + paired t-test.
- [ ] **Chẩn đoán D-01:** SASRec ID-only × 5 dataset × {full, tail10} × 5 seed → so với hàng SASRec của paper → chọn chế độ dữ liệu.
- [ ] (Tier 2) GRU4Rec, FMLP; (Tier 3) DiffuRec.

**Gate 2:** ở chế độ dữ liệu đã chọn, SASRec ID-only có N@20 (trung bình 5 seed) lệch ≤ 5% so với paper trên ≥ 4/5 dataset; D-01 chuyển CLOSED.

### Phase 3 — Stage 1 encoder (viết trên Mac, ~3–4 ngày)
- [ ] `prompts.py`: chuỗi prompt chính xác của code (title: không chat template; ảnh: chat template có system prompt mặc định, `add_generation_prompt=False`); variant cho prompt bảng item (D-07); unit test so chuỗi ký tự.
- [ ] `collate.py`: ảnh 112×112 bicubic, `process_vision_info(..., image_patch_size=14)`, ép `padding_side="right"`, chỉ số token cuối = `attention_mask.sum-1`.
- [ ] `embedder.py`: Qwen2.5-VL-3B-Instruct, bf16, LoRA (r16, α32, p0.2) chỉ trên các linear của LM; gradient checkpointing (`use_reentrant=False`) + `enable_input_require_grads`; không truyền `labels`.
- [ ] `losses.py`: `sft`, `wpcl{code,paper}`, `crtr{code,paper}` + flag ablation (`no_penalty`, `no_stopgrad`, `no_crtr`, `crtr_only`) + các penalty variant (`static`, `batch`, `modality_fixed`).
  - Unit test: variant `code` so với bản chép lại công thức của `loss.py` trên tensor ngẫu nhiên (viết lại trong test, không import từ third_party); variant `paper` so với ví dụ tính tay; kiểm tra stop-grad (không có gradient qua margin mạnh).
- [ ] `trainer.py`: AdamW lr 1e-5, 3 epoch, batch 32 (fallback 8 / GradCache — D-03), không scheduler; checkpoint mỗi N bước + resume; log loss, margin theo modality, thống kê Δgap và w, tỉ lệ user yếu text/ảnh; **log gradient dynamics** `cos(g_fusion, g_T/g_V)` mỗi k bước bằng `autograd.grad` trên embedding đã detach (rẻ, phục vụ Fig. 1).
- [ ] `export.py`: bảng item text/ảnh/fused (fp32, kèm sids) + embedding chuỗi user cho valid/test.
- [ ] `evaluate.py`: full ranking (dot/cosine — D-06; mask lịch sử — D-08), kèm sampled-100 để đối chiếu code; xuất TEXT/VISION/FUSION (dropout test) và geometry U/A/S.
- [ ] Flag few-shot `K_u`.
- [ ] Smoke trên Mac: K_u=64, batch 4, 5 bước → export → eval Stage 1 → plug Stage 2. Overfit test: 32 user, loss → ~0.
- [ ] `notebooks/colab_stage1.ipynb` (mount Drive, clone/sync repo, `uv sync --extra cuda`, chạy + resume) và `docker/Dockerfile.cuda`.

**Gate 3:** test xanh; overfit test đạt; chạy trọn pipeline nhỏ end-to-end trên Mac; đo được thời gian/bước trên GPU đích để cập nhật §3.2.

### Phase 4 — Chạy chính (GPU, ~3–6 ngày chạy máy)
Thứ tự: **ML-1M** (nhỏ nhất) → **Beauty** → Toys → Sports → Clothing.

Với mỗi dataset:
1. **VLM_Vanilla** (không train): export → eval Stage 1 → plug SASRec × 5 seed. Đây là phép thử rẻ nhất cho độ trung thực của prompt/embedding/eval: nếu lệch > 10% so với paper thì **dừng lại debug** (§7) trước khi train.
2. **VLM_SFT**: train → export → eval → plug × 5 seed.
3. **VLM2Rec**: train → export → eval → plug × 5 seed.
4. (Chỉ Beauty + ML-1M) Variant `paper` của WPCL/CRTR, batch 8 so với 32 → đóng D-03/D-04/D-05.

**Gate 4 (mỗi dataset):** VLM_Vanilla lệch ≤ 10%; VLM2Rec > VLM_SFT > VLM_Vanilla ở cả 2 setting; N@20 của VLM2Rec và VLM2Rec_plug lệch ≤ 5% (Beauty, ML-1M) hoặc ≤ 10% (Toys, Sports, Clothing).

### Phase 5 — Phân tích (GPU nhẹ + Mac, ~2–3 ngày)
- [ ] Fig. 1: dropout test (TEXT/VISION/FUSION) + gradient dynamics (Beauty, ML; sau đó Toys, Clothing).
- [ ] Tab. geometry U/S (lấy từ embedding đã có — không tốn thêm).
- [ ] Ablation: w/o WPCL (D-21), WPCL w/o stop-grad, w/o w_pen, w/o CRTR × {Beauty, ML} × 2 setting.
- [ ] WPCL-only (plug, có dropout test); penalty variants (Clothing, ML).
- [ ] Few-shot K_u ∈ {256, 512, 1024} trên Toys/Beauty (rẻ — cũng là cách dò siêu tham số nhanh).
- [ ] Backbone GRU4Rec/FMLP cho plug.
- [ ] Cold-start (nhóm theo số tương tác train) và cross-domain (Clothing→Beauty, Sports→Clothing: chỉ cần inference với model nguồn).
- [ ] (Tier 3) User-level Δgap & tương quan Pearson; OGM-GE/AGM; hard negatives.

### Phase 6 — Baseline mở rộng (tuỳ chọn)
- **Tier 2:** LLM_Vanilla/LLM_SFT (Qwen2.5-3B, nhánh text-only của cùng pipeline — D-13); BERT (bert-large-uncased) và CLIP (clip-vit-large-patch14) đóng băng + mean-pool.
- **Tier 3:** SLIM/SLIM+, LLMEmb, LLM2Rec, I-LLMRec, MLLM-MSR, NoteLLM-2, VLM_Prompt, DiffuRec, hybrid (UniSRec, TedRec, M3SRec, HM4SR), VLM khác (InternVL3, LLaVA, Qwen2.5-VL-7B/32B), Qwen3-VL-Embedding. Mặc định dùng số trong paper làm tham chiếu; chỉ làm khi thật cần.

### Phase 7 — Tổng hợp (~1 ngày)
- [ ] `docs/RESULTS.md` đầy đủ: paper vs ours vs Δ%, kèm run_id.
- [ ] Báo cáo tái hiện: phần nào khớp, phần nào không và vì sao; các quyết định trong DECISIONS đã đóng.
- [ ] Vẽ lại các hình chính; script chạy lại toàn bộ từ đầu.

---

## 6. Ma trận thí nghiệm & ưu tiên

| ID | Thí nghiệm | Tier | Chạy ở | Chi phí | Đối chiếu |
|---|---|---|---|---|---|
| E1 | SASRec ID-only × 5 DS × 2 regime × 5 seed | 1 | Mac | thấp | hàng SASRec |
| E2 | VLM_Vanilla × 5 DS (Stage 1 + plug) | 1 | GPU (ML có thể trên Mac) | thấp | Tab. chính |
| E3 | VLM_SFT × 5 DS | 1 | GPU | ~25 h* | Tab. chính |
| E4 | VLM2Rec × 5 DS | 1 | GPU | ~25 h* | Tab. chính |
| E5 | Dropout test + geometry (từ E2–E4) | 1 | Mac/GPU | ~0 | Fig. 1, Tab. U/S |
| E6 | Gradient dynamics (log trong E3/E4) | 1 | — | ~0 | Fig. 1 phải |
| E7 | Kiểm chứng variant code/paper, batch 8/32 (Beauty, ML) | 1 | GPU | ~12 h* | D-03/04/05 |
| E8 | Ablation 4 variant (Beauty, ML) | 2 | GPU | ~24 h* | Fig. ablation |
| E9 | LLM_Vanilla / LLM_SFT | 2 | GPU | ~15 h* | Tab. chính |
| E10 | Few-shot K_u | 2 | GPU/Mac | thấp | Fig. few-shot |
| E11 | GRU4Rec / FMLP plug | 2 | Mac | thấp | Tab. backbone |
| E12 | BERT / CLIP | 2 | Mac/GPU | thấp | Tab. chính |
| E13 | WPCL-only, penalty variants | 2 | GPU | ~10 h* | Tab. 4, 6 |
| E14 | Cold-start / cross-domain | 2 | Mac/GPU | thấp | Fig. 4 |
| E15+ | Tier 3 (xem Phase 6) | 3 | — | cao | — |

\* giờ GPU hạng RTX 3090, ước lượng.

**MVP (mốc đầu tiên nên nhắm tới):** E1 + E2 + E3 + E4 + E5 trên **ML-1M và Beauty** — đây là 2 dataset paper dùng cho mọi phân tích.

---

## 7. Thang debug khi số lệch (từ rẻ đến đắt)
1. **Dữ liệu:** Checkpoint A khớp chưa? Chế độ `full` hay `tail10` (D-01)? SASRec ID-only khớp chưa?
2. **Giao thức đánh giá:** mask lịch sử (D-08/D-10), full vs sampled, định nghĩa metric, valid/test đúng vị trí.
3. **VLM_Vanilla:** chuỗi prompt, chat template/system prompt, padding side, chỉ số token cuối, kích thước ảnh 112, chuẩn hoá L2, fused = tổng, dot vs cosine (D-06), prompt bảng item (D-07).
4. **VLM_SFT:** dạng loss (D-12), batch size (D-03), lr, số epoch, module LoRA, dtype.
5. **VLM2Rec:** variant WPCL/CRTR (D-04/D-05), τ_WPCL, τ_CRTR, λ (D-11) — dò bằng few-shot K_u=1024 trước.
6. **Plug:** `emb_fusion` (D-09), bias của adapter, max_len, dropout, CE, thứ tự sid của bảng item.

---

## 8. Rủi ro & cách giảm thiểu
| Rủi ro | Giảm thiểu |
|---|---|
| Paper và code mâu thuẫn | Flag `variant`, chẩn đoán rẻ (SASRec, Vanilla), sổ DECISIONS |
| URL ảnh Amazon chết dần | Đã kiểm 25/25 OK; tải sớm và lưu lại; item lỗi bị loại giống code |
| Không có TMDB key / thiếu poster ML-1M | Xin key (miễn phí); dự phòng tra theo imdbId; ghi số phim thiếu |
| OOM ở batch 32 trên GPU 24 GB | Gradient checkpointing; GradCache để giữ 32 negative; fallback batch 8 và ghi lại |
| Colab ngắt phiên | Checkpoint theo bước + resume; lưu lên Drive |
| MPS thiếu op / sai số | `PYTORCH_ENABLE_MPS_FALLBACK=1`; chỉ báo số từ CUDA |
| Thư viện đổi hành vi Qwen2.5-VL | Khoá phiên bản như code tác giả |
| Tie timestamp, seed → nhiễu ±1–3% | 5 seed cho Stage 2; báo mean ± std |
| Ngân sách GPU | Làm MVP trước; Tier 2/3 chỉ khi MVP khớp |

---

## 9. Tiêu chí thành công
- **Mức 1 (bắt buộc):** đúng xu hướng — VLM2Rec > VLM_SFT > VLM_Vanilla ở cả 2 setting; SFT làm rộng gap TEXT–VISION, VLM2Rec thu hẹp gap và FUSION > TEXT.
- **Mức 2 (mục tiêu):** N@20 và H@20 của SASRec, VLM_Vanilla, VLM_SFT, VLM2Rec, VLM2Rec_plug lệch ≤ 5% (tương đối) trên Beauty & ML-1M, ≤ 10% trên Toys/Sports/Clothing.
- **Mức 3 (mở rộng):** VLM2Rec vs VLM_SFT có ý nghĩa thống kê (paired t-test, 5 seed, p < 0.05); thứ tự các ablation khớp paper.

---

## 10. Cách làm việc với Claude qua nhiều session
- `CLAUDE.md` được Claude Code tự nạp mỗi session và **import `docs/STATUS.md`** → mỗi session bắt đầu với đúng trạng thái hiện tại.
- `docs/STATUS.md` (ngắn, ≤ 60 dòng): phase hiện tại, việc tiếp theo, đang chờ gì từ bạn.
- `docs/SESSION_LOG.md`: nhật ký chỉ ghi thêm, mỗi session một mục.
- `docs/DECISIONS.md`: mọi lựa chọn mơ hồ + bằng chứng; không bàn lại mục đã CLOSED.
- `docs/RESULTS.md`: số liệu paper vs ours, mỗi số kèm run_id.
- `docs/PAPER_NOTES.md`, `docs/REFERENCE_CODE_NOTES.md`: kiến thức tham chiếu → không cần đọc lại paper/code mỗi lần.
- Lệnh tắt: `/session-end` (cập nhật STATUS + SESSION_LOG + RESULTS + DECISIONS trước khi dừng), `/log-run <run_dir>` (đưa một run vào RESULTS).
- Bạn chỉ cần mở session mới và nói "tiếp tục" — Claude đọc STATUS và làm mục "Việc tiếp theo".

---

## 11. Cần bạn quyết định / cung cấp
1. **TMDB API key** (miễn phí, đăng ký tại themoviedb.org) để lấy poster ML-1M — hoặc đồng ý dùng bộ poster có sẵn của bên thứ ba.
2. **Lựa chọn GPU cho Stage 1:** Colab Pro/Pro+ (đủ cho MVP) hay thuê GPU (RunPod/Vast; khuyến nghị cho toàn bộ) — và ngân sách.
3. Có **khởi tạo git** và tạo **repo GitHub private** không (Colab/GPU thuê sẽ clone từ đó).
4. Phạm vi: dừng ở MVP (ML-1M + Beauty) hay làm đủ 5 dataset + Tier 2.

## 12. Lịch trình ước tính
| Phase | Thời gian làm | Ghi chú |
|---|---|---|
| 0 | 0.5 ngày | |
| 1 | 1–2 ngày | chủ yếu chờ tải ảnh/meta |
| 2 | 1–2 ngày | |
| 3 | 3–4 ngày | |
| 4 | 3–6 ngày | chủ yếu thời gian chạy GPU |
| 5 | 2–3 ngày | |
| 7 | 1 ngày | |
| **Tổng (không tính Phase 6)** | **~2–3 tuần** | |

# DECISIONS — Sổ quyết định & điểm mơ hồ

> Mỗi điểm mà paper và code chính thức không khớp (hoặc paper không nói) có một mục ở đây.
> Trạng thái: **OPEN** (chưa chọn) · **PROVISIONAL** (đã chọn tạm, chờ kiểm chứng) · **CLOSED** (đã kiểm chứng bằng run).
> Quy tắc: mỗi lựa chọn có ≥2 phương án phải tồn tại dưới dạng flag trong config. Khi đóng một mục, ghi run_id làm bằng chứng.
> Chi tiết kỹ thuật: `docs/PAPER_NOTES.md`, `docs/REFERENCE_CODE_NOTES.md` (bảng C1–C12).

## Nguyên tắc chung
- **P-1. Mục tiêu là khớp số trong paper.** Số trong paper do code của tác giả sinh ra, nên khi paper và code mâu thuẫn, mặc định theo **code** (`variant: code`); phương án theo **paper** được chạy làm đối chứng. Ngoại lệ: chỗ mà code rõ ràng không thể sinh ra số trong paper (vd. eval Stage 1 dùng sampled negatives) → theo paper.
- **P-2.** Không đóng một mục nếu chưa có run chứng minh. "Hợp lý" không phải bằng chứng.

## Danh sách

| ID | Câu hỏi | Phương án | Bằng chứng hiện có | Lựa chọn | Trạng thái | Cách kiểm chứng |
|---|---|---|---|---|---|---|
| D-01 | Chế độ dữ liệu: chuỗi đầy đủ hay cắt 10 item cuối? | (a) `full`: 5-core + join meta, không cắt (khớp bảng thống kê) · (b) `tail10`: code (tail 10 → lọc ảnh → 5-core lại) | Bảng thống kê khớp (a); nhưng ML avg len 165 không thể đưa vào VLM ở batch 32; SASRec ML-1M trong paper H@10=0.1263, thấp hơn nhiều so với SASRec trên ML-1M đầy đủ (thường ~0.2+) → nghiêng về (b) | (b) `tail10` | PROVISIONAL | Phase 2: SASRec ID-only trên cả 2 chế độ × 5 dataset, so với hàng SASRec của paper. Chọn chế độ khớp ±5% |
| D-02 | Mẫu huấn luyện Stage 1 | (a) 1 mẫu/user (history `seq[:-2]`, target `seq[-2]`) · (b) mọi prefix | Code dùng (a); thời gian 93 phút/epoch Beauty ≈ 4 mẫu/s khớp (a); trục x hình gradient (~12k bước Beauty) chưa giải thích được | (a) | PROVISIONAL | So thời gian/epoch và kết quả Beauty |
| D-03 | Batch size Stage 1 | 32 (paper) · 8 (code/README) | Paper ghi rõ 32 trên RTX 3090 | 32; fallback 8 (hoặc GradCache giữ 32 negatives) nếu OOM | PROVISIONAL | Chạy ML-1M với 32 và 8, so N@20 |
| D-04 | Công thức WPCL | `code` (NT-Xent đối xứng 2B, margin theo cặp, w=1+σ(.8)·softplus(σ(.5)·Δ), log w cộng vào logit âm) · `paper` (1 chiều, margin theo user = pos−mean(neg), w=1+softplus(Δ)) | Xem REFERENCE_CODE_NOTES C1–C4, C12 | `code` | PROVISIONAL | Beauty + ML: chạy cả 2 variant, so với paper |
| D-05 | Nhiệt độ CRTR | `code`: softmax(s)/τ (τ thành hệ số loss) · `paper`: softmax(s/τ) | C5 | `code` | PROVISIONAL | Như D-04 |
| D-06 | Hàm similarity khi xếp hạng / trong loss fused | dot product của tổng chưa chuẩn hoá (code) · cosine (paper) | C6 | dot (code) | PROVISIONAL | Đánh giá lại cùng embeddings bằng cả 2 (rẻ, không cần train lại) |
| D-07 | Chuỗi prompt | Chuỗi chính xác của code; prompt cho bảng item: dạng target lúc train (`0). {title}`, `0)`+ảnh) · dạng lúc embed của code (`0){title}`, `{sid})`+ảnh) | C8 | Chuỗi code; bảng item dùng **dạng target lúc train** (nhất quán) | PROVISIONAL | VLM_Vanilla (không cần train) với cả 2 dạng; giữ dạng nào khớp paper hơn |
| D-08 | Eval Stage 1 (VLM2Rec setting) | full ranking có mask lịch sử · full ranking không mask · 100 negatives (code) | Số của paper (N@20≈0.07) chỉ hợp lý với full ranking | full ranking + mask lịch sử; log cả bản không mask và bản 100-neg | PROVISIONAL | VLM_Vanilla: so 3 cách với paper |
| D-09 | Bảng item ở plug setting | `modality_only` (chỉ W·e_i, d=128 — đúng chữ trong paper) · `concat` [ID ‖ W·e_i] (mặc định của code) | Paper: encoder không có CF (BERT, CLIP, VLM_Vanilla) "đều thua SASRec ID-only" ở plug → khó xảy ra nếu có ID embedding | `modality_only` | PROVISIONAL | VLM_Vanilla plug trên Beauty/ML với cả 2 mode; paper: 0.0463 < SASRec 0.0483 (Beauty) |
| D-10 | Mask lịch sử ở Stage 2 | gán điểm = 0 (code) · = −inf | Code gán 0 | 0 (giống code), log thêm −inf | PROVISIONAL | So số trên 1 dataset |
| D-11 | Siêu tham số τ_WPCL, τ_CRTR, λ | mặc định code 0.05/0.1/0.1 · giá trị tốt nhất đọc từ hình | Hình hyperparameter: τ_WPCL=0.05 tốt nhất; ML thích λ=1.0, τ_CRTR=0.01 | mặc định code cho mọi dataset | PROVISIONAL | Nếu lệch >5%: grid nhỏ bằng few-shot K_u=1024 |
| D-12 | Loss của baseline VLM_SFT | cùng khung NT-Xent của code với w≡1 · Eq. 2 một chiều | Cần so sánh "chỉ khác objective" (paper O2) | NT-Xent w≡1 (khớp D-04) | PROVISIONAL | Đổi cùng lúc với D-04 |
| D-13 | Model cho LLM_* | Qwen2.5-3B base · Qwen2.5-3B-Instruct | Paper chỉ ghi "Qwen2.5-3B" | Instruct (cùng họ với VLM Instruct) | OPEN | Tier 2 |
| D-14 | Poster ML-1M | TMDB API qua `links.csv` (ml-25m/ml-latest để phủ đủ ID) · bộ poster có sẵn của bên thứ ba | Paper: TMDB + links file | TMDB API (cần API key của bạn) | OPEN | Đếm số phim có poster ≈ 3,397 |
| D-15 | Metadata Amazon | `meta_<Category>.json.gz` theo từng ngành · `metadata.json.gz` toàn bộ (3.36 GB, code dùng) | Cùng nguồn snap.stanford.edu | theo ngành (nhẹ hơn) | PROVISIONAL | Checkpoint A khớp bảng thống kê |
| D-16 | Thứ tự các tương tác cùng timestamp | sort ổn định (giữ thứ tự file) | Amazon timestamp theo ngày → nhiều tie; không thể khớp tuyệt đối với tác giả | sort ổn định | CLOSED (chấp nhận nhiễu) | — |
| D-17 | "5 seeds" | 5 seed cho cả VLM · chỉ cho backbone SR | Paper không nói rõ; train VLM 5 lần rất tốn | VLM 1 seed (42); Stage 2 5 seed (42–46); thêm seed VLM nếu dư ngân sách | PROVISIONAL | — |
| D-18 | Độ chính xác số | bf16 (CUDA Ampere+, MPS) | T4 không có bf16 | bf16; không dùng T4 cho Stage 1 | CLOSED | — |
| D-19 | LoRA & optimizer | theo code: q,k,v,o,gate,up,down ở LM; vision + merger đóng băng; AdamW wd 0.01, lr 1e-5, không scheduler/clipping, 3 epoch, dùng checkpoint epoch cuối | Paper khớp phần nêu ra | theo code | PROVISIONAL | — |
| D-20 | β, τ "learnable" trong WPCL code | hằng số σ(0.8), σ(0.5) (hành vi thực tế trên CUDA) · learnable | C12 | hằng số | PROVISIONAL | — |
| D-21 | Ablation "w/o WPCL" là gì? | CRTR + SFT không có penalty? · CRTR + loss khác? | Kết quả (0.0652 Beauty) > Vanilla nhiều → vẫn có tín hiệu CF; "w/o w_pen" đã là SFT+CRTR | OPEN | OPEN | Thử: (i) CRTR + NT-Xent chỉ trên text/vision riêng; (ii) CRTR + SFT 1 chiều; chọn cái khớp |
| D-22 | Độ dài history tối đa trong prompt | do D-01 quyết định (tail10 → ≤8 lúc train, ≤9 lúc test) · cắt riêng khi build prompt nếu dùng `full` | — | theo D-01 | PROVISIONAL | — |

## Lịch sử thay đổi
- 2026-11-25 — Tạo sổ, D-01…D-22 (session 1).

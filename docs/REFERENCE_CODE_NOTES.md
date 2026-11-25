# REFERENCE_CODE_NOTES — official anonymous release

> Copy: `third_party/vlm2rec_anon/` (read-only; never import or run it). Fetched 2026-11-25 via
> `https://anonymous.4open.science/api/repo/VLM2Rec/files/` and `.../file/<path>` (the `/r/` web URL returns 401 without JS).
> The release states: "Baselines and ablation variants used in the paper are **not** included". MIT "for review purposes".
> Tested by authors with torch 2.7, transformers 4.55, peft 0.17, single RTX 3090.

## 1. Pipeline overview
| Stage | Files | What it does |
|---|---|---|
| Data | `datasets/download_data.sh`, `download_meta.sh`, `preprocess_data.py`, `build_item_meta.py`, `download_img_pool.py` | Amazon 2014 5-core reviews + full `metadata.json.gz` (3.36 GB) → `processed/<DS>.txt` (`uid sid1 sid2 ...`), `<DS>_item_meta.csv` (`sid,title,imgurl`), `<DS>_image_pool/<sid>.jpg` |
| Stage 1 | `main.py` → `train.py` → `evaluate.py`; `models.py`, `dataset.py`, `loss.py`, `utils.py`, `metrics.py` | LoRA-train Qwen2.5-VL with WPCL(+CRTR), then embed all items, then sampled eval |
| Stage 2 | `cf_baselines/*` | SASRec on frozen item embeddings (+ ID embeddings by default), full-ranking eval |

## 2. Data preprocessing (`datasets/preprocess_data.py`)
Order of operations in `preprocess_sequential` (as called from `__main__`):
1. Load reviews; keep `overall > 0` (i.e. all). Columns user/item/timestamp (`unixReviewTime`, day granularity → many ties).
2. `iterative_5core` (users ≥5 and items ≥5, repeat until stable).
3. `sort_values([user, time])` (multi-key → lexsort, stable w.r.t. file order).
4. Inner-merge with metadata rows having non-null `asin,title,imUrl` (`imUrl` renamed `imgurl`).
5. **`max_seq_len=10`** → `groupby(user).tail(10)` (keep the 10 most recent items per user).
6. `filter_df_items_with_valid_images`: GET each URL (after rewrite `ecx.images-amazon.com` → `https://m.media-amazon.com`), keep items with HTTP 200 + `image/*`.
7. `reapply_core_after_trim=True` → `iterative_5core` again.
8. Remap ids in order of first appearance (`df[col].unique()` on the user-sorted frame); users `1..U`, items `1..I` (strings).
9. Write `<DS>.txt` and `<DS>_item_meta.csv` (`sid,asin,title,imgurl` — `item_id` column also kept).
- Steps 5–7 only run when `max_seq_len is not None`. With `max_seq_len=None` the output would match the paper's statistics table exactly in spirit (5-core + title/image join, no re-filter).
- `build_item_meta.py`: replaces `,` and `;` in titles with spaces (in-place on the CSV). No HTML-unescape.
- `download_img_pool.py`: downloads `<sid>.jpg` (RGB re-save, original size).
- **ML-1M is not supported by the released script** (it would merge ML item ids with Amazon `asin` → empty). TMDB poster scraping is not included. Our own ML pipeline is required.
- We verified on 2026-11-25: snap.stanford.edu 5-core + `meta_<Category>.json.gz` files are online; 25/25 sampled Beauty `imUrl`s return 200 after the `m.media-amazon.com` rewrite.

## 3. Stage 1 — prompts & inputs (exact strings)
- Text history prompt (`dataset.py::make_prompt`, **no chat template**, raw `processor(text=...)`):
  `"Describe this item title sequence in one word., " + ", ".join(f"{idx}). {title}")` → e.g. `Describe this item title sequence in one word., 0). A, 1). B`
- Text target prompt (training): `"Describe this item title sequence in one word., 0). {title}"`
- Text item prompt at embedding time (`utils.generate_item_embeddings`): `"Describe this item title sequence in one word., 0){title}"` ← **differs** from training target (no `. `).
- Image prompts use `processor.apply_chat_template(messages, add_generation_prompt=False)` → Qwen default system prompt ("You are a helpful assistant.") is included; text ends with `<|im_end|>\n`, so the pooled "last token" is the trailing `\n`.
  - History: user content = `[text "Describe this item image sequence in one word.,", text "0)", image, text "1)", image, ...]` (train uses no trailing space; `main.py`/`evaluate.py` set `"...word., "` with a trailing space — minor inconsistency).
  - Target (training): `[prompt, "0)", image]`.
  - Item table (embedding time): `[prompt, f"{sid})", image]` ← **differs** (uses the global item id, not `0)`).
- Images: `PIL.open().convert("RGB").resize((112,112), BICUBIC)` → Qwen2.5-VL: 8×8 patches of 14px → 2×2 merge → **16 visual tokens per image**. `process_vision_info(m, image_patch_size=14)` (needs a recent `qwen_vl_utils`).
- Padding: right (Qwen2Tokenizer default; `padding_side` is null in `tokenizer_config.json`). Pool index = `attention_mask.sum(1) - 1`. `truncation=True` (text max_length 1000).
- Embedding: `outputs.hidden_states[-1]` at the pool index → `F.normalize`. Fused = `text + image` (no re-normalization).
- Training data (`SRDataset`): **one sample per user**: history `seq[:-2]`, target `seq[-2]`. Users with `len(seq) < 3` skipped. Test: history `seq[:-1]`, target `seq[-1]`. No validation split / no model selection (fixed epochs).
- Few-shot: `--data_sample_K K` samples K users (train only).

## 4. Stage 1 — model & optimization
- `AutoModelForCausalLM` registered to `Qwen2_5_VLForConditionalGeneration`; bf16; `device_map="auto"`; `use_cache=False`.
- LoRA r=16, α=32, dropout=0.2, targets auto-picked = `q,k,v,o,gate,up,down` **in the language model only** (vision tower & merger excluded). `task_type=CAUSAL_LM`.
- Gradient checkpointing `use_reentrant=False` + `enable_input_require_grads()`.
- AdamW (default wd=0.01), lr 1e-5, **no scheduler, no warmup, no grad clipping**. Defaults: epochs 3, **batch_size 8** (README example also 8; paper says 32).
- Unused: `w_text`, `w_image` parameters are optimized but never used in the training fusion (dead code). `labels` are passed for targets → HF also computes an unused LM loss.
- Checkpoints saved at epoch 1 and final epoch; evaluation loads the final epoch adapter.

## 5. Stage 1 — losses (`loss.py`) vs paper
### WPCL (code)
```
z = cat(seq, target)  (2B×d) for fused / text / image
sim_* = z @ z.T / temp_cf            # dot products; fused vectors are NOT unit-norm
mask removes self and the positive pair → each of 2B anchors has 2B-2 negatives (both sequences and targets)
margin_m = pos_m[:,None] - sim_m_neg      # per (anchor, negative) PAIR, in logit units (÷ temp_cf)
weak/strong chosen per PAIR; gap = strong.detach() - weak
w = 1 + sigmoid(beta) * softplus(sigmoid(tau) * gap)     # beta=0.8, tau=0.5 "learnable"
logits = [pos_f, neg_f + log(w)] → CrossEntropy (label 0)   # symmetric NT-Xent over 2B anchors
```
- `self.beta = nn.Parameter(...).to(device)` → on CUDA `.to()` returns a plain tensor, so beta/tau are **not registered and never updated** → effectively constants: w = 1 + 0.690·softplus(0.622·gap). (On CPU they would be registered and learnable — device-dependent behaviour.)
### CRTR (code)
```
sim_t = text_seq @ text_target.T ; sim_i = image_seq @ image_target.T      # B×B, candidates = in-batch targets
P_t = softmax(sim_t) / temp_aux ; P_i = softmax(sim_i) / temp_aux           # temperature applied AFTER softmax
loss = 0.5 * mean(sum P_t (log P_t - log P_i) + sum P_i (log P_i - log P_t))
```
- Mathematically = **(1/temp_aux) × symmetric KL of softmax(sim) at temperature 1** → `temp_aux` acts as a loss multiplier, not a temperature. Effective weight = λ/τ_CRTR (defaults 0.1/0.1 = 1.0).

### Discrepancy table (paper ↔ code)
| # | Aspect | Paper | Official code | Impact |
|---|---|---|---|---|
| C1 | Contrastive form | one-directional seq→items, negatives = other users' targets | symmetric NT-Xent over 2B, negatives include other sequences | different gradients; more negatives |
| C2 | Margin | per user, pos − mean(neg), cosine units | per (anchor,neg) pair, logit units (÷τ) | gap scale ×1/τ=×20 |
| C3 | Penalty | w = 1 + softplus(Δ) per user | w = 1 + σ(0.8)·softplus(σ(0.5)·Δ) per pair | magnitude & granularity |
| C4 | Penalty application | multiply Σ negatives by w | add log w to each negative logit (per-pair w) | equivalent only if w is per-anchor |
| C5 | CRTR temperature | softmax(s/τ) | softmax(s)/τ | τ becomes a loss weight |
| C6 | Similarity | cosine | dot product of un-normalized fused sums (norm ∈ [0,2]) | ranking scale |
| C7 | Batch size | 32 | 8 (default & README) | #negatives 31 vs 14 |
| C8 | Prompts | "text sequence", "in one word," "0)" | "title sequence", "in one word.,", "0)." ; item-table prompts differ from train-time target prompts | embedding drift train↔inference |
| C9 | Sequence length | not stated; stats table shows full sequences (ML avg 165) | tail(10) + image validation + re-5-core | **data regime** |
| C10 | Stage-1 eval | full ranking | 100 sampled negatives (`num_neg_sample=100`); full-ranking branch is buggy (targets set to 0) | paper numbers look like full ranking (N@20≈0.07) |
| C11 | Plug item table | adapter only, d=128 | default `emb_fusion=concat` → [ID emb(128) ‖ W·e_i(128)], hidden 256 | could change plug numbers a lot |
| C12 | Learnable β/τ in WPCL | not mentioned | intended learnable, effectively frozen on CUDA | — |

## 6. Stage 2 — `cf_baselines`
- Data: same `<DS>.txt`. Train = all prefixes of `seq[-(max_len+2):-2]` (max_len 50); valid = `seq[:-1]` (target `seq[-2]`); test = full seq (target `seq[-1]`). Left-padding with 0.
- SASRec: 2 layers, 2 heads, inner 256, GELU, dropout 0.5 (hidden & attn), LayerNorm eps 1e-12, init N(0,0.02), learned positional emb, causal mask with −10000.
- Item embedding modes (`abstract_model.Emb`): `lm_embed=False` → ID only; `lm_embed=True` + `emb_fusion` ∈ {`add`, `concat` (default in README/run_backbone), `modality_only`}. Projector = `nn.Linear(lm_dim, hidden, bias=False)`; frozen table loaded from `item_emb_*.pt` (`fused_embeddings`, concatenated in file order; row 0 = padding zeros). **Item order must match sid 1..I** — the loader relies on file naming, not sids.
- Loss: `ce` (default) = full softmax over all items (logits = seq_out · item_table^T); `bce` with 1 sampled negative.
- Optim: Adam lr 1e-3, β (0.9, 0.999), wd 0, batch 256, epochs ≤300, early stopping patience 10 on valid **N@20** (`scores[-1:]`).
- Eval: full ranking when `num_negatives=0`; history items masked by setting their score to **0** (not −inf) using a CSR matrix of `seq[:-2]` (valid) / `seq[:-1]` (test); metrics HR/NDCG @ {1,5,10,15,20}; top-20 via argpartition.
- `main.py` parses `--key value` pairs with `eval()` (unsafe but irrelevant for us).

## 7. What is missing from the release (we must build)
- ML-1M preprocessing & TMDB posters; per-category metadata handling (they use the 3.36 GB global file).
- Full-ranking Stage-1 evaluator (with history masking option).
- VLM_Vanilla / VLM_SFT / LLM_* baselines, BERT/CLIP, other text/multimodal baselines.
- GRU4Rec / FMLP / DiffuRec backbones; hybrid models.
- Ablations (stop-grad off, no penalty, no CRTR, penalty variants), OGM-GE/AGM, gradient-dynamics logging, geometry metrics, dropout test, user-level analysis, cold-start/cross-domain.
- Multi-seed runner & paired t-test.

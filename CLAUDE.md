# CLAUDE.md — VLM2Rec reproduction

## Mission
Re-implement arXiv:2603.17450v3 "VLM2Rec: Resolving Modality Collapse in Vision-Language Models for Sequential Recommendation" (Kim et al., WWW '27) and reproduce its reported numbers as closely as possible. Master plan: `PLAN.md` (Vietnamese).

## Language
- Talk to the user in **Vietnamese**. Code, identifiers, comments, docstrings, commit messages: English.
- User-facing docs (`PLAN.md`, `docs/STATUS.md`, `docs/SESSION_LOG.md`, `docs/DECISIONS.md`, `docs/RESULTS.md`): Vietnamese. Technical references (`docs/PAPER_NOTES.md`, `docs/REFERENCE_CODE_NOTES.md`): English.

## Session protocol
**Start:** the current state is imported at the bottom of this file (`docs/STATUS.md`). Continue from its "Việc tiếp theo" unless the user says otherwise. Before touching a method detail, look it up in `docs/PAPER_NOTES.md` / `docs/REFERENCE_CODE_NOTES.md` instead of recalling it; before making a choice, check `docs/DECISIONS.md` and don't reopen CLOSED items.
**End (or after any milestone):** run `/session-end` — update `docs/STATUS.md` (keep it ≤ 60 lines), append to `docs/SESSION_LOG.md`, put new numbers in `docs/RESULTS.md`, update `docs/DECISIONS.md`, tick boxes in `PLAN.md`.

## Source-of-truth hierarchy
1. Paper: `paper/arxiv_2603.17450v3/` (TeX v3) — what the method *is*.
2. Official anonymous release: `third_party/vlm2rec_anon/` — how it was *actually run* (data prep, prompts, image size, losses, Stage 2).
3. `docs/DECISIONS.md` — how conflicts between 1 and 2 are resolved. Default rule P-1: follow the code, keep the paper version behind a `variant: code|paper` flag.

## Hard rules
- `paper/` and `third_party/` are read-only references. Never modify `third_party/`, and never import it from `src/vlm2rec/`; re-implement there (tests may transcribe reference formulas).
- Exception (branch `explore/official-code`, user-approved 2026-11-25): the official code may be *run* to audit it, but only from a working copy outside the repo (Colab `/content/work/official`) via `tools/official/*` and `notebooks/official/*`. Do not run it on the Mac without asking.
- No number goes into `docs/RESULTS.md` without a run dir `runs/<run_id>/` holding `config.yaml` + `metrics.json` + `env.json`. Paper numbers are labelled "paper".
- Every knob where paper ≠ code is a config flag; record which value each run used.
- Deterministic preprocessing (stable sorts, fixed seeds); log content hashes of processed data.
- Mac is for development, data, Stage 2, smoke tests and few-shot only. Never launch a > 30 min Stage-1 job on the Mac without asking. Final numbers come from CUDA.
- Docker on macOS has no GPU/MPS → the container (`docker/Dockerfile.cuda`) is for CUDA hosts only.
- Never commit `data/`, `artifacts/`, checkpoints, embeddings, `.env`, or API keys.
- Ask before: spending money (cloud GPU), using the user's API keys/tokens (TMDB, HF, W&B), deleting data/artifacts/runs, `git push`, creating remote repos.

## Environment
- Mac: Apple M4 Pro, 24 GB unified memory, macOS 26, `uv` 0.10, Docker 29, ~255 GB free disk. No conda.
- Python 3.11 via `uv`. Pins (match authors): `torch==2.7.0`, `transformers==4.55.0`, `peft==0.17.0`, `qwen-vl-utils` (must support `process_vision_info(..., image_patch_size=14)`).
- Device order cuda > mps > cpu; on Mac export `PYTORCH_ENABLE_MPS_FALLBACK=1`. bf16 everywhere (no T4).
- Commands (fill in as they are built):
  - tests: `uv run pytest -q`
  - lint: `uv run ruff check src tests`

## Repo map
- `PLAN.md` — phases, gates, experiment matrix, debug ladder.
- `docs/` — STATUS, SESSION_LOG, DECISIONS, RESULTS, PAPER_NOTES, REFERENCE_CODE_NOTES.
- `src/vlm2rec/{utils,data,stage1,stage2,eval,analysis}` — implementation (to be built).
- `tools/official/`, `notebooks/official/`, `docs/official_code/{GUIDE_COLAB,AUDIT}.md` — audit of the official release on Colab (branch `explore/official-code`); results land in `docs/official_code/results/`.
- `configs/`, `scripts/`, `tests/`, `notebooks/`, `docker/`.
- `data/`, `artifacts/`, `runs/` — local outputs (mostly gitignored).

## Key facts (details and citations in docs/)
- Embedding = L2-normalized last-token hidden state of the final layer of Qwen2.5-VL-3B-Instruct; title and image sequences are encoded with separate prompts; fused = e^T + e^V (no re-normalization).
- Official code: images resized to 112×112 (16 visual tokens); image prompts go through the chat template (default system prompt, `add_generation_prompt=False`), title prompts do not; right padding.
- Official preprocessing keeps only the last 10 items per user (then image check + re-5-core); the paper's stats table is pre-truncation. Decided empirically in Phase 2 (D-01).
- Stage 1 training: one sample per user (history `seq[:-2]` → target `seq[-2]`), LoRA r16/α32/p0.2 on LM linears only, AdamW 1e-5, 3 epochs, batch 32 (paper) vs 8 (code).
- WPCL/CRTR in code differ from the paper equations (pairwise margins in logit units, constant σ(0.8)/σ(0.5) coefficients, CRTR temperature applied after softmax). See REFERENCE_CODE_NOTES §5.
- Paper's Stage-1 numbers are full ranking; the code's Stage-1 eval uses 100 sampled negatives → we implement full ranking.
- Stage 2: SASRec 2L/2H, d=128, max_len 50, dropout 0.5, CE over all items, Adam 1e-3, bs 256, patience 10 on valid N@20, history masked (score 0). Paper = adapter only (`modality_only`); code default = `concat` with ID embeddings (D-09).
- Hyperparameters: τ_WPCL 0.05, τ_CRTR 0.1, λ 0.1 (code defaults).

## Current state
@docs/STATUS.md

# tools/official — run & audit the official VLM2Rec release

Helpers for branch `explore/official-code`. They **call** the official code from a working copy
(e.g. `/content/work/official`, copied from `third_party/vlm2rec_anon/`); the repo copy is never modified,
and nothing in `src/vlm2rec/` imports it.

| Script | Purpose | Needs GPU |
|---|---|---|
| `prep_amazon.py` | Official Amazon preprocessing (parallel network I/O) + "Checkpoint A" stats → `prep_report_<DS>.json` | no |
| `audit_probes.py` | Runtime probes P1–P8 confirming the static review → `probes_<DS>.json` | no (P4 is device-dependent) |
| `fullrank_eval.py` | Full-ranking Stage-1 eval (+ dropout test, geometry, sampled-100), trained or vanilla → `fullrank_<DS>_<tag>.json`; `--selftest` checks the ranking maths | yes (except `--selftest`) |
| `run_stage2.py` | Official `cf_baselines/main.py` with official defaults; modes `id_only / modality_only / concat / add` → `stage2_<DS>.json` | recommended |
| `summarize.py` | All JSONs → Markdown table vs paper (`paper_targets.py`) | no |

Usage: see the Colab notebooks in `notebooks/official/` and `docs/official_code/GUIDE_COLAB.md`.
Local check (no GPU): `uv run --no-project --with torch python tools/official/fullrank_eval.py --selftest`.

---
description: Record a finished run (runs/<run_id>/) into docs/RESULTS.md with paper comparison
argument-hint: <run_dir or run_id> [more run dirs...]
---

Record the following run(s) into `docs/RESULTS.md`: $ARGUMENTS

For each run:
1. Read `runs/<run_id>/config.yaml`, `metrics.json`, `env.json`. If `metrics.json` is missing or the run did not finish, stop and tell the user — never log partial or guessed numbers.
2. Add a row to "## 7. Nhật ký run" (stage, dataset, method, variant flags that differ from defaults, device, key metrics).
3. If it maps to a comparison table (§2–§6), fill the "Ours" cell, compute Δ% against the paper value from `docs/PAPER_NOTES.md`, and reference the run_id. For multi-seed Stage-2 results use mean ± std over seeds and list the seed run_ids.
4. Flag in your reply any metric with |Δ| above the PLAN.md §9 tolerance and suggest the next step from the PLAN.md §7 debug ladder.
5. If the run is evidence for a decision in `docs/DECISIONS.md`, update that entry.

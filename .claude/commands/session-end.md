---
description: Wrap up the session — update STATUS, SESSION_LOG, RESULTS, DECISIONS and PLAN checkboxes
---

Wrap up the current work session for the VLM2Rec reproduction. Do all of the following, writing in Vietnamese (as the docs are):

1. `docs/SESSION_LOG.md`: append a new "## Session N — <today's date>" entry using the template at the top of the file (goal, what was done, discoveries/surprises, runs created with their key numbers, decision changes, unfinished work). Number N = previous session + 1.
2. `docs/STATUS.md`: rewrite it to reflect the state right now — current phase, "Đã xong" (only the most relevant items), "Việc tiếp theo" (ordered, concrete, first item executable immediately), "Đang chờ từ bạn", the most important open decisions. Keep it ≤ 60 lines; move older detail to SESSION_LOG instead.
3. `docs/RESULTS.md`: add every run finished this session (only runs with `runs/<run_id>/metrics.json`) to the run log and to the relevant comparison tables with Δ%.
4. `docs/DECISIONS.md`: update status/choice/evidence (with run_id) for any decision that moved; add new ambiguities found; add a line to "Lịch sử thay đổi".
5. `PLAN.md`: tick completed checkboxes; if estimates (time, GPU hours) were measured, update §3.2.
6. If `CLAUDE.md` "Commands" or "Key facts" became outdated, fix them.
7. Report to the user in Vietnamese: a short summary of what changed and the next step.

Do not invent numbers; if something was not run, say so.

$ARGUMENTS

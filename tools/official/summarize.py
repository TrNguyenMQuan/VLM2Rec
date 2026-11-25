#!/usr/bin/env python3
"""Summarize audit results of the official code into a Markdown report (paper vs official run).

Reads from --audit_dir:
  prep_report_<DS>.json   (prep_amazon.py)
  probes_<DS>.json        (audit_probes.py)
  fullrank_<DS>_*.json    (fullrank_eval.py)
  stage2_<DS>.json        (run_stage2.py)
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_targets import DATA_STATS, DROPOUT_N20, METRICS, TARGETS  # noqa: E402


def pct(ours, paper):
    if ours is None or paper in (None, 0):
        return "—"
    return f"{(ours - paper) / paper * 100:+.1f}%"


def fmt(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def load(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except FileNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit_dir", required=True)
    ap.add_argument("--data_name", default="Beauty")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ds, d = args.data_name, args.audit_dir
    L = [f"# Official-code audit — {ds}", ""]

    prep = load(os.path.join(d, f"prep_report_{ds}.json"))
    L += ["## 1. Data", ""]
    if prep:
        tgt = DATA_STATS.get(ds, {})
        a = prep["checkpoint_A"]
        L += ["| Stage | Users | Items | Interactions | Avg len |", "|---|---|---|---|---|"]
        L.append(f"| Paper table | {tgt.get('users')} | {tgt.get('items')} | {tgt.get('interactions')} | {tgt.get('avg_len')} |")
        for key, name in (("raw_reviews_5core_file", "Raw 5-core file"), ("after_5core", "After 5-core (re-run)"),
                          ("checkpoint_A", "**Checkpoint A** (5-core + meta join)"),
                          ("official_output", "Official output (tail10 + img check + re-5-core)")):
            s = prep.get(key)
            if s:
                L.append(f"| {name} | {s['users']} | {s['items']} | {s['interactions']} | {s['avg_len']} |")
        L += ["", f"Checkpoint A vs paper: users {pct(a['users'], tgt.get('users'))}, items {pct(a['items'], tgt.get('items'))}, "
                  f"interactions {pct(a['interactions'], tgt.get('interactions'))}.",
              f"Image URL check: {prep.get('image_url_check')}; images: {prep.get('images', {}).get('missing_after')} missing "
              f"of {prep.get('images', {}).get('n_items_in_meta')}.", ""]
    else:
        L += ["(no prep report)", ""]

    probes = load(os.path.join(d, f"probes_{ds}.json"))
    L += ["## 2. Runtime probes", ""]
    if probes:
        L += ["| Probe | Status | Message |", "|---|---|---|"]
        for p in probes:
            msg = str(p["msg"]).replace("|", "\\|").replace("\n", " ")
            L.append(f"| {p['id']} | {p['status']} | {msg[:400]} |")
        L.append("")
    else:
        L += ["(no probes)", ""]

    L += ["## 3. VLM2Rec setting (Stage 1, full ranking, FUSION|masked|dot)", "",
          "| Run (tag) | paper_key | users | " + " | ".join(f"{m} ours / paper (Δ)" for m in METRICS) + " | sampled-100 N@20 |",
          "|---|---|---|" + "---|" * len(METRICS) + "---|"]
    drop_rows = []
    for f in sorted(glob.glob(os.path.join(d, f"fullrank_{ds}_*.json"))):
        r = load(f)
        tgt = TARGETS.get(r.get("paper_key"), {}).get(ds, {})
        cells = [f"{fmt(r['main'][m])} / {fmt(tgt.get(m))} ({pct(r['main'][m], tgt.get(m))})" for m in METRICS]
        L.append(f"| {r['tag']} | {r.get('paper_key')} | {r['n_users']}/{r['n_users_all']} | " + " | ".join(cells)
                 + f" | {fmt(r['sampled100_FUSION_dot']['N@20'])} |")
        dp = r["dropout_N@20_masked_dot"]
        pd_ = DROPOUT_N20.get(ds, {}).get(r.get("paper_key"), {})
        drop_rows.append(f"| {r['tag']} | {fmt(dp['TEXT'])} / {fmt(pd_.get('TEXT'), 3)} | {fmt(dp['VISION'])} / "
                         f"{fmt(pd_.get('VISION'), 3)} | {fmt(dp['FUSION'])} / {fmt(pd_.get('FUSION'), 4)} | {dp['gap_pct']}% |")
        unm = r["all"]["FUSION|unmasked|dot"]["N@20"]
        cos = r["all"]["FUSION|masked|cosine"]["N@20"]
        drop_rows_extra = f"<!-- {r['tag']}: N@20 unmasked={unm:.4f}, cosine={cos:.4f} -->"
        drop_rows.append(drop_rows_extra)
    L += ["", "### Dropout test (N@20, masked, dot): ours / paper (Fig. 1, approx.)", "",
          "| Run | TEXT | VISION | FUSION | Gap (T−V)/V |", "|---|---|---|---|---|"]
    L += [r for r in drop_rows if not r.startswith("<!--")]
    L += [""] + [r for r in drop_rows if r.startswith("<!--")] + [""]

    L += ["## 4. Plug setting (Stage 2, official SASRec, full ranking)", "",
          "| Label | mode | seeds | epochs run | " + " | ".join(f"{m} ours / paper (Δ)" for m in METRICS) + " |",
          "|---|---|---|---|" + "---|" * len(METRICS)]
    st2 = load(os.path.join(d, f"stage2_{ds}.json")) or []
    for r in st2:
        tgt = TARGETS.get(r.get("paper_key"), {}).get(ds, {})
        ep = ",".join(str(s.get("epochs_run")) for s in r["per_seed"])
        cells = []
        for m in METRICS:
            sd = r["std"].get(m, 0.0)
            ours = f"{r['mean'][m]:.4f}" + (f"±{sd:.4f}" if len(r["per_seed"]) > 1 else "")
            cells.append(f"{ours} / {fmt(tgt.get(m))} ({pct(r['mean'][m], tgt.get(m))})")
        L.append(f"| {r['label']} | {r['mode']} | {len(r['per_seed'])} | {ep} | " + " | ".join(cells) + " |")
    L += ["", "Paper keys: sasrec = ID-only SASRec row; *_plug = VLM2Rec_plug setting rows of the main table.", ""]

    text = "\n".join(L)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        open(args.out, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the OFFICIAL Stage-2 backbone (cf_baselines/main.py) with the official default hyper-parameters.

Defaults are read (without importing) from cf_baselines/run_backbone.py (`Params` + first value of
`search_space`), then overridden per mode:
  id_only        -> lm_embed False            (pure ID SASRec; paper row "SASRec")
  concat         -> lm_embed True, emb_fusion concat         (official default: [ID || W e_i])
  modality_only  -> lm_embed True, emb_fusion modality_only  (paper text: adapter W e_i only)
  add            -> lm_embed True, emb_fusion add

Item embeddings (item_emb_*.pt written by the official generate_item_embeddings) are copied to
<official>/datasets/item_emb/<DS>/<emb_name>/ which is where the official loader looks.
The final test metrics are parsed from the official log and appended to --out_json.

Example:
  python tools/official/run_stage2.py --official_dir /content/work/official --data_name Beauty \
      --mode modality_only --emb_dir /content/work/official/results/vlm_item_emb/Beauty/vanilla/epoch_0/Qwen_Qwen2.5-VL-3B-Instruct \
      --emb_name vanilla --label VLM_Vanilla_plug_modality_only --paper_key vanilla_plug \
      --seeds 42 --out_json /content/drive/MyDrive/VLM2Rec_work/audit_results/stage2_Beauty.json
"""

import argparse
import ast
import glob
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

MODES = ("id_only", "concat", "modality_only", "add")


def read_official_defaults(run_backbone_py):
    tree = ast.parse(open(run_backbone_py, encoding="utf-8").read())
    params, space = {}, {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Params":
            for st in node.body:
                if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                    params[st.target.id] = ast.literal_eval(st.value)
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "search_space" for t in node.targets):
            space = ast.literal_eval(node.value)
    if not params or not space:
        raise RuntimeError("could not parse Params/search_space from run_backbone.py")
    merged = dict(params)
    merged.update({k: v[0] for k, v in space.items()})
    return merged


def link_embeddings(emb_dir, dst):
    src_files = sorted(glob.glob(os.path.join(emb_dir, "item_emb_*.pt")))
    if not src_files:
        raise FileNotFoundError(f"no item_emb_*.pt in {emb_dir}")
    os.makedirs(dst, exist_ok=True)
    existing = glob.glob(os.path.join(dst, "item_emb_*.pt"))
    if len(existing) != len(src_files):
        for f in existing:
            os.remove(f)
        for f in src_files:
            shutil.copy2(f, dst)
    return len(src_files)


def parse_test_metrics(log_path):
    text = open(log_path, encoding="utf-8").read()
    idx = text.rfind("Test Score")
    if idx < 0:
        raise RuntimeError(f"'Test Score' not found in {log_path}")
    dicts = re.findall(r"\{'Epoch'.*?\}", text[idx:])
    if not dicts:
        raise RuntimeError(f"no metric dict after 'Test Score' in {log_path}")
    d = ast.literal_eval(dicts[-1])
    m = {
        "H@5": float(d["HR@5"]), "H@10": float(d["HR@10"]), "H@20": float(d["HR@20"]),
        "N@5": float(d["NDCG@5"]), "N@10": float(d["NDCG@10"]), "N@20": float(d["NDCG@20"]),
    }
    n_epochs = len(re.findall(r"'rec_loss'", text))
    return m, n_epochs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--official_dir", required=True)
    ap.add_argument("--data_name", required=True)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--emb_dir", default=None, help="dir with item_emb_*.pt (not needed for id_only)")
    ap.add_argument("--emb_name", default=None, help="sub-folder name under datasets/item_emb/<DS>/")
    ap.add_argument("--model_name", default="sasrec")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--label", default=None)
    ap.add_argument("--paper_key", default="none")
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--gpu_id", default="0")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="extra overrides, e.g. max_seq_length=50")
    args = ap.parse_args()

    off = os.path.abspath(args.official_dir)
    cf = os.path.join(off, "cf_baselines")
    base = read_official_defaults(os.path.join(cf, "run_backbone.py"))

    emb_name = args.emb_name or ("idonly" if args.mode == "id_only" else "emb")
    n_emb_files = None
    if args.mode != "id_only":
        if not args.emb_dir:
            ap.error("--emb_dir is required unless --mode id_only")
        n_emb_files = link_embeddings(args.emb_dir, os.path.join(off, "datasets", "item_emb", args.data_name, emb_name))

    label = args.label or f"{args.model_name}_{args.mode}_{emb_name}"
    per_seed = []
    for seed in args.seeds:
        p = dict(base)
        p.update({
            "model_name": args.model_name,
            "emb_model": emb_name,
            "data_name": args.data_name,
            "lm_embed": args.mode != "id_only",
            "emb_fusion": args.mode if args.mode != "id_only" else base.get("emb_fusion", "concat"),
            "seed": seed,
            "gpu_id": args.gpu_id,
            "do_eval": False,
        })
        for kv in args.set:
            k, v = kv.split("=", 1)
            p[k] = v
        cmd = [sys.executable, "main.py"]
        for k, v in p.items():
            cmd += [f"--{k}", str(v)]
        print(f"[stage2] {label} seed={seed}\n  cd {cf} && {' '.join(cmd)}", flush=True)
        t0 = time.time()
        subprocess.run(cmd, cwd=cf, check=True)
        folder = os.path.join(cf, str(p["output_dir"]), f"{emb_name}_{args.model_name}_{args.data_name}")
        logs = [f for f in glob.glob(os.path.join(folder, "*.log")) if os.path.getmtime(f) >= t0 - 1]
        if not logs:
            raise RuntimeError(f"no new log in {folder}")
        log_path = max(logs, key=os.path.getmtime)
        metrics, n_epochs = parse_test_metrics(log_path)
        rec = {"seed": seed, "metrics": metrics, "epochs_run": n_epochs, "log": log_path,
               "seconds": round(time.time() - t0, 1)}
        print(f"[stage2] {label} seed={seed} -> {metrics}", flush=True)
        per_seed.append(rec)

    keys = per_seed[0]["metrics"].keys()
    mean = {k: statistics.mean(r["metrics"][k] for r in per_seed) for k in keys}
    std = {k: (statistics.stdev(r["metrics"][k] for r in per_seed) if len(per_seed) > 1 else 0.0) for k in keys}
    record = {
        "label": label, "paper_key": args.paper_key, "data_name": args.data_name, "mode": args.mode,
        "emb_name": emb_name, "emb_dir": args.emb_dir, "n_emb_files": n_emb_files,
        "params": {k: str(v) for k, v in base.items()}, "overrides": args.set,
        "seeds": args.seeds, "per_seed": per_seed, "mean": mean, "std": std,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(json.dumps({"label": label, "mean": mean, "std": std}, indent=2))
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        data = []
        if os.path.exists(args.out_json):
            data = json.load(open(args.out_json, encoding="utf-8"))
        data = [r for r in data if r.get("label") != label] + [record]
        json.dump(data, open(args.out_json, "w", encoding="utf-8"), indent=2)
        print(f"[stage2] appended to {args.out_json}")


if __name__ == "__main__":
    main()

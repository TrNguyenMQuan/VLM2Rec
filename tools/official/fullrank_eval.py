#!/usr/bin/env python3
"""Full-ranking Stage-1 evaluation ("VLM2Rec setting") for the OFFICIAL encoder.

Why: the official evaluate.py ranks the target against 100 sampled negatives, while the paper reports
full ranking over all items. This script reuses the official code (models.VLEmbedder, dataset.SRDataset
prompt builders, utils._build_messages / process_inputs_eval / generate_item_embeddings) and ranks ALL items.

Reported variants (paper-comparable one = FUSION|masked|dot):
  modality  TEXT / VISION / FUSION  -> sequence embedding e_u^T, e_u^V or e_u^T+e_u^V vs the FUSED item table
                                       (= the paper's dropout test, Fig. 1)
  mask      masked / unmasked       -> history items (except the target) removed from the ranking or not
  score     dot / cosine            -> official code uses dot products of un-normalized fused sums
Also: official-style sampled metric (target vs 100 random negatives) and optional geometry (U, S).

Modes:
  * trained:  --adapter_path <.../hf_model> --item_emb_dir <results/vlm_item_emb/...>
  * vanilla:  no --adapter_path; item embeddings are generated with the official generate_item_embeddings
              using the base model (tag given by --tag).
  * --selftest: checks the ranking maths on random data (no model, CPU ok).
"""

import argparse
import glob
import json
import math
import os
import random
import re
import sys
import time
from types import SimpleNamespace

KS = (5, 10, 20)


# ----------------------------------------------------------------------------- ranking maths
def rank_metrics(seq, items, targets, hist, ks=KS, mask=True, cosine=False, chunk=512):
    """seq (U,d), items (I,d), targets LongTensor (U,) 0-based rows, hist: list of lists of 0-based rows."""
    import torch
    import torch.nn.functional as F

    if cosine:
        seq = F.normalize(seq, dim=-1)
        items = F.normalize(items, dim=-1)
    n = seq.size(0)
    acc = {f"H@{k}": 0.0 for k in ks}
    acc.update({f"N@{k}": 0.0 for k in ks})
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        sc = seq[s:e] @ items.T
        tgt = targets[s:e].to(sc.device)
        if mask:
            rows, cols = [], []
            for r, (h, t) in enumerate(zip(hist[s:e], tgt.tolist())):
                for x in h:
                    if x != t:
                        rows.append(r)
                        cols.append(x)
            if rows:
                sc[torch.tensor(rows, device=sc.device), torch.tensor(cols, device=sc.device)] = float("-inf")
        tscore = sc.gather(1, tgt.view(-1, 1))
        rank = (sc > tscore).sum(dim=1)
        for k in ks:
            hit = rank < k
            acc[f"H@{k}"] += hit.sum().item()
            acc[f"N@{k}"] += torch.where(hit, 1.0 / torch.log2(rank.float() + 2.0),
                                         torch.zeros_like(rank, dtype=torch.float)).sum().item()
    return {k: v / n for k, v in acc.items()}


def sampled_metrics(fused_seq, fused_items, targets, full_seqs, n_neg=100, ks=KS, seed=42):
    """Official-style: target vs n_neg random items not in the user's whole sequence (dot product)."""
    import torch

    rng = random.Random(seed)
    n_items = fused_items.size(0)
    acc = {f"H@{k}": 0.0 for k in ks}
    acc.update({f"N@{k}": 0.0 for k in ks})
    for u in range(fused_seq.size(0)):
        pos = set(full_seqs[u])
        negs = set()
        while len(negs) < min(n_neg, n_items - len(pos)):
            c = rng.randrange(n_items)
            if c not in pos:
                negs.add(c)
        cand = torch.tensor([int(targets[u])] + sorted(negs), device=fused_items.device)
        sc = fused_items[cand] @ fused_seq[u]
        rank = int((sc > sc[0]).sum().item())
        for k in ks:
            if rank < k:
                acc[f"H@{k}"] += 1
                acc[f"N@{k}"] += 1.0 / math.log2(rank + 2)
    return {k: v / fused_seq.size(0) for k, v in acc.items()}


def geometry(user_vec, item_vec, targets, n_neg=100, n_pairs=50000, seed=42):
    """Wang & Isola uniformity + paper's separability S = A_neg / A_pos (vectors L2-normalized)."""
    import torch
    import torch.nn.functional as F

    g = torch.Generator(device="cpu").manual_seed(seed)
    u = F.normalize(user_vec.float().cpu(), dim=-1)
    it = F.normalize(item_vec.float().cpu(), dim=-1)
    nu, ni = u.size(0), it.size(0)
    # for unit vectors ||a-b||^2 = 2 - 2 a.b ; chunked to keep memory small
    a_pos = (2 - 2 * (u * it[targets.cpu()]).sum(-1)).mean().item()
    neg_idx = torch.randint(0, ni, (nu, n_neg), generator=g)
    neg_sum = 0.0
    for s in range(0, nu, 1024):
        dots = torch.einsum("bd,bkd->bk", u[s:s + 1024], it[neg_idx[s:s + 1024]])
        neg_sum += (2 - 2 * dots).sum().item()
    a_neg = neg_sum / (nu * n_neg)
    pa = torch.randint(0, nu, (n_pairs,), generator=g)
    pb = torch.randint(0, ni, (n_pairs,), generator=g)
    d2 = torch.cat([2 - 2 * (u[pa[s:s + 8192]] * it[pb[s:s + 8192]]).sum(-1) for s in range(0, n_pairs, 8192)])
    uni = torch.log(torch.exp(-2 * d2).mean()).item()
    return {"U": uni, "A_pos": a_pos, "A_neg": a_neg, "S": a_neg / (a_pos + 1e-8)}


def selftest():
    import torch

    torch.manual_seed(0)
    U, I, d = 300, 500, 16
    seq, items = torch.randn(U, d), torch.randn(I, d)
    targets = torch.randint(0, I, (U,))
    hist = [random.Random(u).sample(range(I), 7) for u in range(U)]
    for mask in (False, True):
        for cos in (False, True):
            fast = rank_metrics(seq, items, targets, hist, mask=mask, cosine=cos, chunk=64)
            s = torch.nn.functional.normalize(seq, dim=-1) if cos else seq
            it = torch.nn.functional.normalize(items, dim=-1) if cos else items
            ref = {f"H@{k}": 0.0 for k in KS}
            ref.update({f"N@{k}": 0.0 for k in KS})
            for u in range(U):
                sc = (it @ s[u]).clone()
                if mask:
                    for x in hist[u]:
                        if x != int(targets[u]):
                            sc[x] = float("-inf")
                order = torch.argsort(sc, descending=True).tolist()
                r = order.index(int(targets[u]))
                for k in KS:
                    if r < k:
                        ref[f"H@{k}"] += 1 / U
                        ref[f"N@{k}"] += (1 / math.log2(r + 2)) / U
            for k in ref:
                assert abs(ref[k] - fast[k]) < 1e-6, (mask, cos, k, ref[k], fast[k])
    smp = sampled_metrics(seq, items, targets, [h + [int(t)] for h, t in zip(hist, targets)], n_neg=100)
    assert 0.0 <= smp["N@20"] <= 1.0
    geo = geometry(seq, items, targets, n_pairs=2000)
    assert geo["U"] < 0 and geo["S"] > 0
    print("selftest OK: rank_metrics matches brute-force argsort (mask x cosine); sampled/geometry run")


# ----------------------------------------------------------------------------- official model glue
def load_item_table(item_emb_dir):
    import torch

    def idx(p):
        return int(re.findall(r"\d+", os.path.basename(p))[-1])

    files = sorted(glob.glob(os.path.join(item_emb_dir, "item_emb_*.pt")), key=idx)
    if not files:
        raise FileNotFoundError(f"no item_emb_*.pt in {item_emb_dir}")
    parts = [torch.load(f, map_location="cpu") for f in files]
    sids = torch.cat([p["sids"] for p in parts]).long()
    tab = {k: torch.cat([p[k] for p in parts]).float() for k in ("fused_embeddings", "text_embeddings", "image_embeddings")}
    expected = torch.arange(1, sids.numel() + 1)
    if not torch.equal(sids, expected):
        raise RuntimeError("item table rows are not sids 1..N in order")
    return tab, len(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--official_dir")
    ap.add_argument("--data_name")
    ap.add_argument("--data_path", default=None, help="default <official_dir>/datasets/processed")
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--vlm_cache_dir", default="./pretrained_vlm", help="relative to official_dir (official default)")
    ap.add_argument("--adapter_path", default=None, help="LoRA hf_model dir; omit for VLM_Vanilla")
    ap.add_argument("--item_emb_dir", default=None, help="official item_emb_*.pt dir; omit to generate")
    ap.add_argument("--tag", default="vanilla", help="name for generated item embeddings / output")
    ap.add_argument("--paper_key", default="none", help="vanilla | sft | vlm2rec | none")
    ap.add_argument("--image_prompt", default="Describe this item image sequence in one word., ",
                    help="main.py/evaluate.py value (train.py alone uses no trailing space)")
    ap.add_argument("--batch_size", type=int, default=16, help="users per forward pass")
    ap.add_argument("--item_batch_size", type=int, default=8, help="official generate_item_embeddings batch")
    ap.add_argument("--max_users", type=int, default=0, help="0 = all test users")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--geometry", action="store_true")
    ap.add_argument("--save_user_emb", default=None)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.official_dir or not args.data_name:
        ap.error("--official_dir and --data_name are required (or use --selftest)")

    import pandas as pd
    import torch
    from tqdm import tqdm

    t0 = time.time()
    off = os.path.abspath(args.official_dir)
    data_path = os.path.abspath(args.data_path or os.path.join(off, "datasets", "processed"))
    out_json = os.path.abspath(args.out_json) if args.out_json else None
    save_user_emb = os.path.abspath(args.save_user_emb) if args.save_user_emb else None
    item_emb_dir = os.path.abspath(args.item_emb_dir) if args.item_emb_dir else None
    adapter_path = os.path.abspath(args.adapter_path) if args.adapter_path else None
    os.chdir(off)  # official code writes ./results and caches ./pretrained_vlm relative to its root
    sys.path.insert(0, off)
    import utils as U  # official
    from dataset import SRDataset  # official
    from models import VLEmbedder  # official
    from peft import PeftModel
    from transformers import AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_dir = os.path.join(data_path, f"{args.data_name}_image_pool")

    # --- data (mirrors SRDataset mode="test": history seq[:-1], target seq[-1], users with >=3 items)
    users = []
    with open(os.path.join(data_path, f"{args.data_name}.txt"), encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            seq = [int(x) for x in parts[1:]]
            if len(seq) < 3:
                continue
            users.append((parts[0], seq))
    n_all = len(users)
    if args.max_users and args.max_users < n_all:
        keep = sorted(random.Random(args.seed).sample(range(n_all), args.max_users))
        users = [users[i] for i in keep]
    meta = pd.read_csv(os.path.join(data_path, f"{args.data_name}_item_meta.csv"))
    meta["sid"] = meta["sid"].astype(str)
    ds_obj = SRDataset.__new__(SRDataset)  # only use its prompt builders
    ds_obj.id2title = dict(zip(meta["sid"], meta["title"]))

    # --- model (mirrors evaluate.main)
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    embedder = VLEmbedder(
        args.model_name, pooling_type="last", use_lora=False, padding_side="right", target_modules="auto",
        include_xmodal=False, use_gc=False, cache_dir=args.vlm_cache_dir,
        from_pretrained_kwargs={"torch_dtype": torch.bfloat16, "trust_remote_code": True},
    )
    embedder.to(device)
    embedder.eval()
    if adapter_path:
        embedder.model = PeftModel.from_pretrained(embedder.model, adapter_path)
        print(f"[fullrank] loaded adapter {adapter_path}")
    else:
        print("[fullrank] no adapter -> VLM_Vanilla")

    # --- item table
    if item_emb_dir is None:
        ns = SimpleNamespace(data_path=data_path, data_name=args.data_name, load_desc=args.tag, epochs=0,
                             model_name=args.model_name, batch_size=args.item_batch_size, device=device)
        U.generate_item_embeddings(ns, embedder, processor)
        item_emb_dir = os.path.abspath(
            f"./results/vlm_item_emb/{args.data_name}/{args.tag}/epoch_0/{args.model_name.replace('/', '_')}/")
    table, n_files = load_item_table(item_emb_dir)
    n_items = table["fused_embeddings"].size(0)
    print(f"[fullrank] items={n_items} from {item_emb_dir} ({n_files} files); test users={len(users)}/{n_all}")

    # --- encode test histories with the official prompt/collate code
    text_chunks, img_chunks = [], []
    t_enc = time.time()
    for s in tqdm(range(0, len(users), args.batch_size), desc="encode users"):
        batch_users = users[s:s + args.batch_size]
        prompts, msgs = [], []
        for _, seq in batch_users:
            hist = seq[:-1]
            _, hist_str = ds_obj.make_history_string(hist)
            prompts.append(ds_obj.make_prompt(hist_str, mode="input"))
            msgs.append(U._build_messages({"input_ids": hist}, img_dir, args.image_prompt))
        chat = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in msgs]
        imgs = [U.process_vision_info(m, image_patch_size=14)[0] for m in msgs]
        batch = {"input_titles": prompts, "chat_texts": chat, "image_inputs": imgs}
        input_images, input_texts = U.process_inputs_eval(batch, processor)
        with torch.no_grad():
            te = embedder({k: v for k, v in input_texts.items()})
            ve = embedder({k: v for k, v in input_images.items()})
        text_chunks.append(te.float().cpu())
        img_chunks.append(ve.float().cpu())
    enc_sec = time.time() - t_enc
    e_text = torch.cat(text_chunks)
    e_img = torch.cat(img_chunks)
    e_fused = e_text + e_img

    targets = torch.tensor([seq[-1] - 1 for _, seq in users])
    hist_rows = [[x - 1 for x in seq[:-1]] for _, seq in users]
    full_rows = [[x - 1 for x in seq] for _, seq in users]
    items_fused = table["fused_embeddings"].to(device)

    results = {}
    for mod, vec in (("TEXT", e_text), ("VISION", e_img), ("FUSION", e_fused)):
        for mask in (True, False):
            for cosine in (False, True):
                key = f"{mod}|{'masked' if mask else 'unmasked'}|{'cosine' if cosine else 'dot'}"
                results[key] = rank_metrics(vec.to(device), items_fused, targets, hist_rows, mask=mask, cosine=cosine)
    sampled = sampled_metrics(e_fused.to(device), items_fused, targets, full_rows, n_neg=100, seed=args.seed)

    geo = None
    if args.geometry:
        geo = {
            "Text": geometry(e_text, table["text_embeddings"], targets),
            "Vision": geometry(e_img, table["image_embeddings"], targets),
            "Fused(normalized)": geometry(e_fused, table["fused_embeddings"], targets),
        }

    def gap(r_t, r_v):
        return round((r_t - r_v) / r_v * 100, 2) if r_v else None

    main_key = "FUSION|masked|dot"
    dropout = {m: results[f"{m}|masked|dot"]["N@20"] for m in ("TEXT", "VISION", "FUSION")}
    dropout["gap_pct"] = gap(dropout["TEXT"], dropout["VISION"])
    record = {
        "tag": args.tag, "paper_key": args.paper_key, "data_name": args.data_name,
        "adapter_path": adapter_path, "item_emb_dir": item_emb_dir, "model_name": args.model_name,
        "image_prompt": args.image_prompt, "n_users": len(users), "n_users_all": n_all, "n_items": n_items,
        "main_key": main_key, "main": results[main_key], "dropout_N@20_masked_dot": dropout,
        "sampled100_FUSION_dot": sampled, "all": results, "geometry": geo,
        "encode_seconds": round(enc_sec, 1), "seconds": round(time.time() - t0, 1),
        "versions": {"torch": torch.__version__},
    }
    try:
        import peft
        import transformers
        record["versions"].update({"transformers": transformers.__version__, "peft": peft.__version__})
    except Exception:  # noqa: BLE001
        pass

    print(json.dumps({"main": record["main"], "dropout": dropout, "sampled100": sampled}, indent=2))
    if save_user_emb:
        os.makedirs(os.path.dirname(save_user_emb), exist_ok=True)
        torch.save({"uids": [u for u, _ in users], "text": e_text.half(), "image": e_img.half(),
                    "targets": targets}, save_user_emb)
    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        print(f"[fullrank] -> {out_json}")


if __name__ == "__main__":
    main()

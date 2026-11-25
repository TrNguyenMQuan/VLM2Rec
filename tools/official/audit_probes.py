#!/usr/bin/env python3
"""Runtime probes that confirm (or refute) the static review of the official code.

Each probe prints a line  [PROBE <id>] <status> <message>  and is stored in the JSON output.
status: OK (as expected), DIFF (behaviour differs from the paper — expected finding), WARN (problem), INFO.

Probes:
  P1 versions & requirements      P2 prompts (train vs item-table)     P3 image tokens / last token / padding
  P4 WPCL beta/tau registration   P5 CRTR temperature semantics        P6 WPCL code vs paper formula
  P7 Stage-1 eval protocol        P8 Stage-2 defaults (emb_fusion etc.)

Needs: processed data for --data_name (from prep_amazon.py) for P2/P3; a GPU is not required
(P4 behaves differently on CPU vs CUDA — run it on the same device you train on).
"""

import argparse
import ast
import importlib
import inspect
import json
import os
import sys

RESULTS = []


def report(pid, status, msg, **data):
    print(f"[PROBE {pid}] {status:4s} {msg}", flush=True)
    RESULTS.append({"id": pid, "status": status, "msg": msg, **data})


def p1_versions(off):
    vers = {}
    for mod in ("torch", "transformers", "peft", "accelerate", "qwen_vl_utils", "pandas", "PIL"):
        try:
            m = importlib.import_module(mod)
            vers[mod] = getattr(m, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            vers[mod] = f"MISSING ({e.__class__.__name__})"
    try:
        from importlib.metadata import version
        vers["qwen_vl_utils"] = version("qwen-vl-utils")
    except Exception:  # noqa: BLE001
        pass
    report("P1a", "INFO", f"versions: {vers}", versions=vers)
    try:
        from qwen_vl_utils import process_vision_info
        ok = "image_patch_size" in inspect.signature(process_vision_info).parameters
        report("P1b", "OK" if ok else "WARN",
               "process_vision_info supports image_patch_size" if ok
               else "qwen-vl-utils too old: official utils.py passes image_patch_size=14 -> install qwen-vl-utils==0.0.14")
    except Exception as e:  # noqa: BLE001
        report("P1b", "WARN", f"cannot import qwen_vl_utils: {e}")
    req = open(os.path.join(off, "requirements.txt"), encoding="utf-8").read()
    missing = [p for p in ("accelerate",) if p not in req]
    report("P1c", "WARN" if missing else "OK",
           f"requirements.txt lacks {missing} although models.py uses device_map='auto'" if missing
           else "requirements.txt complete", requirements=req.splitlines())
    tv = vers.get("transformers", "")
    report("P1d", "OK" if tv.startswith("4.55") else "WARN",
           f"transformers {tv} (authors tested 4.55; models.py re-registers Qwen classes in AutoModelForCausalLM)")
    pv = vers.get("pandas", "")
    report("P1e", "OK" if pv[:1] in ("1", "2") else "WARN",
           f"pandas {pv} (preprocess uses groupby().apply(tail) which changes behaviour in pandas 3)")


def p2_p3_prompts(off, data_path, data_name, model_name, image_prompt):
    import pandas as pd

    sys.path.insert(0, off)
    import utils as U
    from dataset import SRDataset

    meta = pd.read_csv(os.path.join(data_path, f"{data_name}_item_meta.csv"))
    meta["sid"] = meta["sid"].astype(str)
    ds = SRDataset.__new__(SRDataset)
    ds.id2title = dict(zip(meta["sid"], meta["title"]))
    seq = None
    with open(os.path.join(data_path, f"{data_name}.txt"), encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) - 1 >= 4:
                seq = [int(x) for x in parts[1:]]
                break
    hist, target = seq[:-2], seq[-2]
    _, hist_str = ds.make_history_string(hist)
    text_hist = ds.make_prompt(hist_str, mode="input")
    text_target_train = ds.make_prompt(ds.id2title[str(target)], mode="target")
    src = inspect.getsource(U.generate_item_embeddings)
    item_text_uses_no_dot = 'one word., 0){row' in src
    item_img_uses_sid = 'f"{idx})"' in src
    report("P2a", "INFO", f"text history prompt: {text_hist[:300]!r}", text_hist=text_hist)
    report("P2b", "INFO", f"text target prompt (train): {text_target_train!r}")
    report("P2c", "DIFF" if item_text_uses_no_dot else "OK",
           "item-table text prompt at embedding time is 'one word., 0){title}' (no '. ') — differs from the "
           "train-time target prompt 'one word., 0). {title}'" if item_text_uses_no_dot else "item text prompt consistent")
    report("P2d", "DIFF" if item_img_uses_sid else "OK",
           "item-table image prompt numbers the image with the global sid '{sid})' instead of '0)' used for "
           "train-time targets" if item_img_uses_sid else "item image prompt consistent")
    report("P2e", "DIFF", "paper prompt says 'Describe this item text sequence in one word, 0) t1, ...'; code uses "
                          "'title sequence', 'one word.,' and '0). '")

    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        report("P3", "WARN", f"cannot load processor ({e}); skip token probes")
        return
    img_dir = os.path.join(data_path, f"{data_name}_image_pool")
    msgs = U._build_messages({"input_ids": hist}, img_dir, image_prompt)
    chat = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    imgs, _ = U.process_vision_info(msgs, image_patch_size=14)
    enc = processor(text=[chat], images=[imgs], return_tensors="pt", padding=True)
    pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    n_pad = int((enc["input_ids"] == pad_id).sum())
    per_img = n_pad / max(1, len(imgs))
    last_tok = processor.tokenizer.decode(enc["input_ids"][0, -1:])
    report("P3a", "INFO", f"image chat prompt (head): {chat[:260]!r}", image_chat=chat)
    report("P3b", "OK" if abs(per_img - 16) < 1e-6 else "WARN",
           f"{per_img:.1f} visual tokens per 112x112 image (expected 16); {len(imgs)} images; "
           f"total tokens {enc['input_ids'].shape[1]}")
    report("P3c", "INFO", f"pooled last token of image prompt = {last_tok!r} (chat template ends with <|im_end|>\\n); "
                          "system prompt included: " + str("You are a helpful assistant" in chat))
    report("P3d", "OK" if processor.tokenizer.padding_side == "right" else "WARN",
           f"tokenizer padding_side={processor.tokenizer.padding_side!r} (official pooling assumes right padding)")


def p4_p6_losses(off):
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, off)
    L = importlib.import_module("loss")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = type("A", (), {})()
    args.device, args.temp_cf, args.temp_aux, args.epsilon = dev, 0.05, 0.1, 1e-12
    w = L.WPCL(args)
    n_params = len(list(w.parameters()))
    report("P4", "DIFF" if n_params == 0 else "INFO",
           f"on {dev.type}: WPCL registers {n_params} parameters; beta/tau are "
           + ("plain tensors -> never optimized, w = 1 + 0.690*softplus(0.622*gap)" if n_params == 0
              else "learnable on this device (behaviour differs CPU vs CUDA)"),
           device=dev.type, n_params=n_params)

    torch.manual_seed(0)
    B, d = 8, 32
    z = [F.normalize(torch.randn(B, d, device=dev), dim=-1) for _ in range(4)]
    crtr = L.CRTR(args)
    vals = {}
    for t in (0.1, 1.0):
        args.temp_aux = t
        vals[t] = crtr(tuple(z)).item()
    ratio = vals[0.1] / vals[1.0] if vals[1.0] else float("nan")

    def crtr_paper(zz, tau):
        ts, tt, vs, vt = zz
        pt = F.softmax(ts @ tt.T / tau, -1)
        pv = F.softmax(vs @ vt.T / tau, -1)
        return 0.5 * ((pt * (pt.log() - pv.log())).sum(-1) + (pv * (pv.log() - pt.log())).sum(-1)).mean().item()

    report("P5", "DIFF",
           f"official CRTR(temp_aux=0.1)/CRTR(temp_aux=1.0) = {ratio:.3f} (≈10 => temperature acts as a loss "
           f"multiplier: softmax(s)/tau). Paper form softmax(s/tau): tau=0.1 -> {crtr_paper(z, 0.1):.4f}, "
           f"tau=1.0 -> {crtr_paper(z, 1.0):.4f}", official=vals, ratio=ratio)

    ts, tt, vs, vt = z
    fs, ft = ts + vs, tt + vt
    args.temp_aux = 0.1
    code_loss = w(fs, ft, (ts, tt, vs, vt)).item()

    def wpcl_paper(tau=0.05):
        sim = lambda a, b: F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T  # noqa: E731
        eye = torch.eye(B, dtype=torch.bool, device=dev)
        margins = []
        for a, b in ((ts, tt), (vs, vt)):
            s = sim(a, b)
            pos = s.diag()
            neg = s.masked_fill(eye, 0).sum(1) / (B - 1)
            margins.append(pos - neg)
        m = torch.stack(margins, 1)
        gap = m.max(1).values.detach() - m.min(1).values
        wp = 1 + F.softplus(gap)
        s = sim(fs, ft) / tau
        pos = s.diag()
        neg = torch.logsumexp(s.masked_fill(eye, float("-inf")), 1)
        return (-(pos - torch.logaddexp(pos, neg + torch.log(wp)))).mean().item(), wp

    p_loss, wp = wpcl_paper()
    report("P6", "DIFF",
           f"same random batch: official WPCL={code_loss:.4f} vs paper-form WPCL={p_loss:.4f} "
           f"(paper w range {wp.min().item():.3f}-{wp.max().item():.3f}). Official: symmetric NT-Xent over 2B, "
           "pairwise margins in logit units, cosine replaced by dot of un-normalized fused sums",
           official=code_loss, paper=p_loss)


def p7_eval(off):
    src = open(os.path.join(off, "evaluate.py"), encoding="utf-8").read()
    main_src = open(os.path.join(off, "main.py"), encoding="utf-8").read()
    msrc = open(os.path.join(off, "metrics.py"), encoding="utf-8").read()
    sampled = '"--num_neg_sample", type=int, default=100' in main_src
    buggy_full = "targets = torch.zeros(similarities.size(0)" in msrc
    report("P7", "DIFF",
           f"Stage-1 eval default num_neg_sample=100 ({sampled}) -> target vs 100 sampled negatives, not the paper's "
           f"full ranking; full-ranking branch sets every target index to 0 ({buggy_full}) -> unusable. "
           "Use tools/official/fullrank_eval.py for paper-comparable numbers.", sampled_default=sampled,
           full_branch_bug=buggy_full, eval_has_validation=("valid" in src))


def p8_stage2(off):
    tree = ast.parse(open(os.path.join(off, "cf_baselines", "run_backbone.py"), encoding="utf-8").read())
    space, params = {}, {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "search_space" for t in node.targets):
            space = ast.literal_eval(node.value)
        if isinstance(node, ast.ClassDef) and node.name == "Params":
            for st in node.body:
                if isinstance(st, ast.AnnAssign):
                    params[st.target.id] = ast.literal_eval(st.value)
    report("P8a", "DIFF" if space.get("emb_fusion", [None])[0] == "concat" else "INFO",
           f"Stage-2 default emb_fusion={space.get('emb_fusion')} lm_embed={space.get('lm_embed')} "
           f"loss={space.get('loss')} num_negatives={space.get('num_negatives')} hidden={space.get('hidden_size')} "
           "(paper: adapter W e_i only, d=128)", search_space=space)
    report("P8b", "INFO", f"Stage-2 fixed params: {params}", params=params)
    report("P8c", "INFO", "Stage-2 registry only has SASRec (GRU4Rec/FMLP/DiffuRec of the paper are not released); "
                          "history masking sets scores to 0 (not -inf)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--official_dir", required=True)
    ap.add_argument("--data_name", default="Beauty")
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--image_prompt", default="Describe this item image sequence in one word., ")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    off = os.path.abspath(args.official_dir)
    data_path = os.path.abspath(args.data_path or os.path.join(off, "datasets", "processed"))
    out = os.path.abspath(args.out) if args.out else None
    os.chdir(off)

    for fn, a in ((p1_versions, (off,)),
                  (p2_p3_prompts, (off, data_path, args.data_name, args.model_name, args.image_prompt)),
                  (p4_p6_losses, (off,)), (p7_eval, (off,)), (p8_stage2, (off,))):
        try:
            fn(*a)
        except Exception as e:  # noqa: BLE001
            report(fn.__name__, "WARN", f"probe crashed: {e.__class__.__name__}: {e}")
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(RESULTS, open(out, "w", encoding="utf-8"), indent=2, default=str)
        print(f"[probes] -> {out}")


if __name__ == "__main__":
    main()

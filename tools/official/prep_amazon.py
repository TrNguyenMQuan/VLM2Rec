#!/usr/bin/env python3
"""Prepare one Amazon-2014 dataset with the OFFICIAL VLM2Rec preprocessing code (audit branch).

It imports and calls the official functions from a *working copy* of the release
(<official_dir>/datasets/{preprocess_data,build_item_meta,download_img_pool}.py) and keeps
their semantics. Differences from running the official scripts by hand:
  * per-category metadata (meta_<Category>.json.gz, ~0.1-1 GB) instead of the 3.36 GB global file;
  * image-URL validation and image download run in a thread pool (--workers 1 = official sequential);
  * it also computes "Checkpoint A" statistics (5-core + metadata join, before truncation) to compare
    with the paper's dataset table, and writes a JSON report.

Usage (Colab, CPU runtime is enough):
  python tools/official/prep_amazon.py --official_dir /content/work/official --dataset Beauty \
      --workers 16 --report /content/drive/MyDrive/VLM2Rec_work/audit_results/prep_report_Beauty.json
"""

import argparse
import gzip
import importlib
import io
import json
import os
import shutil
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SNAP = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
# official dataset name -> Amazon 2014 category file stem
CATEGORY = {
    "Beauty": "Beauty",
    "Toys_and_Games": "Toys_and_Games",
    "Sports": "Sports_and_Outdoors",
    "Clothing": "Clothing_Shoes_and_Jewelry",
}
UA = {"User-Agent": "Mozilla/5.0 (research-client)"}


def log(msg):
    print(f"[prep] {msg}", flush=True)


def download(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        log(f"skip download (exists): {dst}")
        return
    log(f"download {url}")
    tmp = dst + ".part"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(tmp, dst)


def gunzip(src, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        log(f"skip gunzip (exists): {dst}")
        return
    with gzip.open(src, "rb") as fi, open(dst + ".part", "wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)
    os.replace(dst + ".part", dst)


def frame_stats(df, user_col="user_id", item_col="item_id"):
    users = int(df[user_col].nunique())
    items = int(df[item_col].nunique())
    inter = int(len(df))
    return {
        "users": users,
        "items": items,
        "interactions": inter,
        "avg_len": round(inter / users, 2) if users else 0.0,
        "sparsity_pct": round((1 - inter / (users * items)) * 100, 3) if users and items else 0.0,
    }


def file_stats(seq_txt):
    users, inter, max_item, min_len = 0, 0, 0, None
    with open(seq_txt, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            seq = [int(x) for x in parts[1:]]
            users += 1
            inter += len(seq)
            max_item = max(max_item, max(seq))
            min_len = len(seq) if min_len is None else min(min_len, len(seq))
    return {
        "users": users,
        "items": max_item,
        "interactions": inter,
        "avg_len": round(inter / users, 2) if users else 0.0,
        "min_len": min_len,
        "users_lt3_items": None,  # filled below
    }


def make_parallel_filter(P, workers):
    """Same semantics as P.filter_df_items_with_valid_images, but concurrent."""
    from tqdm import tqdm

    def _filter(df, item_col="item_id"):
        unique_items = df[[item_col, "imgurl"]].drop_duplicates(item_col)
        items = unique_items[item_col].tolist()
        urls = unique_items["imgurl"].tolist()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            oks = list(tqdm(ex.map(P.is_image_valid, urls), total=len(urls),
                            desc=f"Validate Image URLs (x{workers})"))
        valid = [i for i, ok in zip(items, oks) if ok]
        n_invalid = len(items) - len(valid)
        print(f"\n[IMG FILTER] valid: {len(valid)}, invalid: {n_invalid}")
        df_filtered = df[df[item_col].isin(valid)]
        print(f"[IMG FILTER] df size before={len(df)}, after={len(df_filtered)}")
        _filter.last = {"valid": len(valid), "invalid": n_invalid}
        return df_filtered

    _filter.last = None
    return _filter


def download_images(D, csv_path, image_dir, workers):
    """Same per-item logic as download_img_pool.download_images_from_csv, but concurrent."""
    import csv

    from PIL import Image
    from tqdm import tqdm

    os.makedirs(image_dir, exist_ok=True)
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    local = threading.local()

    def session():
        if not hasattr(local, "s"):
            local.s = D.create_session()
        return local.s

    def one(row):
        save_path = os.path.join(image_dir, f"{row['sid']}.jpg")
        if os.path.exists(save_path):
            return "skip", None
        url = D.fix_url(row["imgurl"])
        try:
            resp = session().get(url, timeout=10)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            img.save(save_path)
            return "ok", None
        except Exception as e:  # noqa: BLE001 - mirror official broad except
            return "fail", f"{row['sid']} {url} {e}"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(tqdm(ex.map(one, rows), total=len(rows), desc=f"Download images (x{workers})"))
    fails = [msg for st, msg in results if st == "fail"]
    missing = [r["sid"] for r in rows if not os.path.exists(os.path.join(image_dir, f"{r['sid']}.jpg"))]
    return {
        "n_items_in_meta": len(rows),
        "downloaded": sum(st == "ok" for st, _ in results),
        "skipped_existing": sum(st == "skip" for st, _ in results),
        "failed": len(fails),
        "failed_examples": fails[:10],
        "missing_after": len(missing),
        "missing_sids_examples": missing[:20],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--official_dir", required=True, help="working copy of third_party/vlm2rec_anon")
    ap.add_argument("--dataset", required=True, choices=sorted(CATEGORY))
    ap.add_argument("--workers", type=int, default=16, help="1 = official sequential behaviour")
    ap.add_argument("--max_seq_len", type=int, default=10,
                    help="official __main__ uses 10; 0 = no truncation (then no image filter / re-5-core either)")
    ap.add_argument("--report", default=None, help="path of the JSON report")
    args = ap.parse_args()

    t0 = time.time()
    ds = args.dataset
    ds_dir = os.path.join(os.path.abspath(args.official_dir), "datasets")
    os.makedirs(os.path.join(ds_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(ds_dir, "processed"), exist_ok=True)
    os.chdir(ds_dir)  # the official code uses paths relative to datasets/
    sys.path.insert(0, ds_dir)
    P = importlib.import_module("preprocess_data")
    B = importlib.import_module("build_item_meta")
    D = importlib.import_module("download_img_pool")

    cat = CATEGORY[ds]
    # 1) raw files (same file names as the official download_data.sh)
    download(f"{SNAP}/reviews_{cat}_5.json.gz", f"./raw/{ds}.json.gz")
    gunzip(f"./raw/{ds}.json.gz", f"./raw/{ds}.json")
    download(f"{SNAP}/meta_{cat}.json.gz", f"./raw/meta_{cat}.json.gz")

    # 2) metadata CSV via the official parser (it always writes ./raw/metadata.csv)
    meta_csv = f"./raw/metadata_{ds}.csv"
    if not os.path.exists(meta_csv):
        log("parse metadata with official load_metadata_df (ast.literal_eval per line) ...")
        P.load_metadata_df(f"./raw/meta_{cat}.json.gz")
        os.replace("./raw/metadata.csv", meta_csv)
    meta = P.load_metadata_csv(meta_csv)

    # 3) statistics before the official truncation (Checkpoint A)
    df = P.load_dataset_df(ds, rating_score=0.0)
    report = {"dataset": ds, "workers": args.workers, "max_seq_len": args.max_seq_len}
    report["raw_reviews_5core_file"] = frame_stats(df)
    df5 = P.iterative_5core(df, "user_id", "item_id")
    report["after_5core"] = frame_stats(df5)
    dfa = df5.merge(meta.rename(columns={"asin": "item_id"}), on="item_id", how="inner")
    report["checkpoint_A"] = frame_stats(dfa)
    log(f"Checkpoint A: {report['checkpoint_A']}")

    # 4) official preprocessing (same call as preprocess_data.py __main__)
    fast_filter = None
    if args.workers > 1:
        fast_filter = make_parallel_filter(P, args.workers)
        P.filter_df_items_with_valid_images = fast_filter
    seq_out = f"./processed/{ds}.txt"
    P.preprocess_sequential(
        df,
        output_path=seq_out,
        user_col="user_id",
        item_col="item_id",
        time_col="timestamp",
        meta_df=meta,
        dataset_name_for_saving=ds,
        max_seq_len=args.max_seq_len if args.max_seq_len > 0 else None,
        reapply_core_after_trim=True,
    )
    if fast_filter is not None:
        report["image_url_check"] = fast_filter.last
    out = file_stats(seq_out)
    with open(seq_out, encoding="utf-8") as f:
        out["users_lt3_items"] = sum(1 for line in f if len(line.split()) - 1 < 3)
    report["official_output"] = out
    log(f"official output: {out}")

    # 5) official build_item_meta (title cleanup + user json)
    item_csv = f"./processed/{ds}_item_meta.csv"
    B.filtering_csv_sep(item_csv, sep=B.SEPERATOR)
    B.build_user_item_json(item_csv, seq_out, f"./processed/{ds}_meta.json")

    # 6) images (official naming: processed/<DS>_image_pool/<sid>.jpg)
    report["images"] = download_images(D, item_csv, f"./processed/{ds}_image_pool", args.workers)
    log(f"images: {report['images']}")
    if report["images"]["missing_after"]:
        log("WARNING: some items have no image -> the official trainer will crash on them "
            "(load_image_from_index has no fallback). Re-run this script to retry.")

    report["seconds"] = round(time.time() - t0, 1)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        log(f"report -> {args.report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

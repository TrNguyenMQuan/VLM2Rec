import sys
import os
import json
from tqdm import tqdm
import numpy as np
import datetime
import pandas as pd
import ast, io, gzip, json
import requests
from PIL import Image

############################################
#  1. SAFE OPEN
############################################

def _safe_open(path, mode='r', encoding='utf-8'):
    if os.path.exists(path):
        return open(path, mode, encoding=encoding)
    dirname, fname = os.path.split(path)
    alt = os.path.join(dirname, fname.lower())
    if os.path.exists(alt):
        return open(alt, mode, encoding=encoding)
    alt2 = os.path.join(dirname, fname.capitalize())
    if os.path.exists(alt2):
        return open(alt2, mode, encoding=encoding)
    raise FileNotFoundError(f"File not found: {path}")

############################################
#  2. READ JSONL
############################################

def _read_jsonl(path):
    records = []
    with _safe_open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            while len(s) and s[-1] in [',']:
                s = s[:-1].rstrip()
            try:
                obj = json.loads(s)
                records.append(obj)
            except json.JSONDecodeError:
                s2 = s.replace('\n', ' ').strip()
                while len(s2) and s2[-1] in [',']:
                    s2 = s2[:-1].rstrip()
                obj = json.loads(s2)
                records.append(obj)
    return records

############################################
#  3. DATASET LOADERS
############################################

def Amazon(dataset_name, rating_score):
    data_file = f'./raw/{dataset_name}.json'
    data = _read_jsonl(data_file)
    df = pd.DataFrame(data)
    df = df[df['overall'] > rating_score]
    df = df[['reviewerID', 'asin', 'unixReviewTime']]
    df.columns = ['user_id', 'item_id', 'timestamp']
    df['timestamp'] = df['timestamp'].astype(int)
    return df

def ML1M(dataset_name, rating_score):
    data_file = f'raw/{dataset_name}/ratings.dat'
    df = pd.read_csv(data_file, sep='::', engine='python')
    df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    df = df[df['rating'] > rating_score]
    df['user_id'] = df['user_id'].astype(int)
    df['item_id'] = df['item_id'].astype(int)
    df['timestamp'] = df['timestamp'].astype(int)
    return df

############################################
#  4. METADATA LOADING
############################################

def _open_maybe_gzip(path):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")

def load_metadata_df(metadata_path="./raw/metadata.json"):
    records = []
    with _open_maybe_gzip(metadata_path) as f:
        for i, line in enumerate(tqdm(f, desc="Reading JSONL-ish")):
            line = line.strip()
            if not line:
                continue
            try:
                obj = ast.literal_eval(line)
            except Exception:
                try:
                    obj = json.loads(line)
                except Exception as e:
                    raise RuntimeError(
                        f"Could not parse line {i+1}. Start: {line[:80]!r}. Error: {e}"
                    )
            if not isinstance(obj, dict):
                raise RuntimeError(f"Line {i+1} did not parse to a dict.")
            records.append(obj)

    meta_df = pd.DataFrame.from_records(records)
    meta_df.to_csv('./raw/metadata.csv')
    return meta_df


def load_metadata_csv(metadata_csv_path='./raw/metadata.csv'):
    meta_df = pd.read_csv(metadata_csv_path)
    meta_df = meta_df[["asin", "title", "imUrl"]]
    meta_df = meta_df.dropna(subset=["asin", "title", "imUrl"])
    meta_df = meta_df.rename(columns={"imUrl": "imgurl"})
    return meta_df[["asin", "title", "imgurl"]]

############################################
#  5. IMAGE VALIDATION + REMOVE INVALID ITEMS
############################################

def fix_url(url):
    if url.startswith("http://ecx.images-amazon.com"):
        url = url.replace("http://ecx.images-amazon.com", "https://m.media-amazon.com")
    if url.startswith("https://ecx.images-amazon.com"):
        url = url.replace("https://ecx.images-amazon.com", "https://m.media-amazon.com")
    return url

def is_image_valid(url):
    try:
        url = fix_url(url)
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if not ctype.startswith("image/"):
            return False
        return True
    except Exception:
        return False

def filter_df_items_with_valid_images(df, item_col='item_id'):
    unique_items = df[[item_col, "imgurl"]].drop_duplicates(item_col)

    valid_items = []
    invalid_items = []

    for _, row in tqdm(unique_items.iterrows(), total=len(unique_items), desc="Validate Image URLs"):
        if is_image_valid(row["imgurl"]):
            valid_items.append(row[item_col])
        else:
            invalid_items.append(row[item_col])

    print(f"\n[IMG FILTER] valid: {len(valid_items)}, invalid: {len(invalid_items)}")

    df_filtered = df[df[item_col].isin(valid_items)]

    print(f"[IMG FILTER] df size before={len(df)}, after={len(df_filtered)}")

    return df_filtered

############################################
#  6. 5-CORE FILTERING
############################################

def iterative_5core(df, user_col='user_id', item_col='item_id'):
    while True:
        prev_len = len(df)
        user_cnt = df[user_col].value_counts()
        item_cnt = df[item_col].value_counts()
        df = df[df[user_col].isin(user_cnt[user_cnt >= 5].index)]
        df = df[df[item_col].isin(item_cnt[item_cnt >= 5].index)]
        if len(df) == prev_len:
            break
    return df

############################################
#  7. STATISTICS
############################################

def statistics(df, user_seqs, user_col='user_id', item_col='item_id'):
    user_num = df[user_col].nunique()
    item_num = df[item_col].nunique()
    interaction_num = len(df)

    if user_num == 0 or item_num == 0:
        print("[WARNING] Empty df → skip statistics")
        return

    avg_seq_len = np.mean([len(seq) for seq in user_seqs.values()])
    sparsity = (1 - interaction_num / (user_num * item_num)) * 100

    user_counts = df[user_col].value_counts()
    item_counts = df[item_col].value_counts()

    print("\n=====Dataset Statistics=====")
    print(f"Total User: {user_num}, Avg User: {user_counts.mean():.2f}, Min Len: {user_counts.min()}, Max Len: {user_counts.max()}")
    print(f"Total Item: {item_num}, Avg Item: {item_counts.mean():.2f}, Min Inter: {item_counts.min()}, Max Inter: {item_counts.max()}")
    print(f"Interaction Num: {interaction_num}, Avg Sequence Length: {avg_seq_len:.1f}, Sparsity: {sparsity:.2f}%")

############################################
#  8. MAIN PREPROCESSING PIPELINE
############################################

def preprocess_sequential(
    df,
    output_path,
    user_col='user_id',
    item_col='item_id',
    time_col='timestamp',
    meta_df=None,
    dataset_name_for_saving=None,
    max_seq_len=None,
    reapply_core_after_trim=False
):
    df = iterative_5core(df, user_col, item_col)
    df = df.sort_values([user_col, time_col])

    if meta_df is not None:
        df = df.merge(
            meta_df.rename(columns={'asin': item_col}),
            on=item_col,
            how='inner'
        )
    else:
        df['title'] = np.nan
        df['imgurl'] = np.nan

    if max_seq_len is not None:
        df = df.groupby(user_col, group_keys=False).apply(lambda x: x.tail(max_seq_len))
        df = filter_df_items_with_valid_images(df, item_col=item_col)
        if reapply_core_after_trim:
            df = iterative_5core(df, user_col, item_col)
        

    if df.empty:
        print("[WARNING] EMPTY after filters")
        return df, {}, {}

    user2id = {u: str(i + 1) for i, u in enumerate(df[user_col].unique())}
    item2id = {i: str(j + 1) for j, i in enumerate(df[item_col].unique())}
    user_seqs = df.groupby(user_col)[item_col].apply(lambda x: [item2id[i] for i in x]).to_dict()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for user, seq in user_seqs.items():
            f.write(f"{user2id[user]} {' '.join(seq)}\n")

    items_meta = df[[item_col, 'title', 'imgurl']].drop_duplicates(item_col)
    items_meta.insert(0, 'sid', items_meta[item_col].map(item2id))

    base = dataset_name_for_saving
    meta_out = os.path.join(os.path.dirname(output_path), f"{base}_item_meta.csv")
    items_meta.to_csv(meta_out, index=False)

    print(f'[SAVE] Sequences -> {output_path}')
    print(f'[SAVE] Item meta  -> {meta_out}')

    statistics(df, user_seqs, user_col, item_col)
    return df, user2id, item2id

############################################
#  9. DATASET LOADER
############################################

def load_dataset_df(dataset_name, rating_score=0.0):
    if dataset_name in [
        "Sports", "Toys_and_Games", "Beauty", "Instruments",
        "Video_Games", "Baby", "beauty",
        "toys_and_games", "instruments", "Clothing", "video_games", "baby"
    ]:
        return Amazon(dataset_name, rating_score)
    elif dataset_name == "ml-1m":
        return ML1M(dataset_name, rating_score)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

############################################
#  10. MAIN
############################################

if __name__ == "__main__":
    dataset_name = sys.argv[1]

    df = load_dataset_df(dataset_name, rating_score=0.0)

    meta = load_metadata_csv("./raw/metadata.csv")

    seq_out = f"./processed/{dataset_name}.txt"
    preprocess_sequential(
        df,
        output_path=seq_out,
        user_col="user_id",
        item_col="item_id",
        time_col="timestamp",
        meta_df=meta,
        dataset_name_for_saving=dataset_name,
        max_seq_len=10,
        reapply_core_after_trim=True
    )

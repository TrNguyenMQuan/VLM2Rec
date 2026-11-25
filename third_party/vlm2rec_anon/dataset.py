import random
from typing import Literal

import pandas as pd
from torch.utils.data import Dataset


class SRDataset(Dataset):
    """Sequential recommendation dataset that emits item-id sequences and titles."""

    def __init__(self, args, mode: Literal["train", "test"] = "train", num_negatives: int = 100):
        assert mode in ["train", "test"]
        self.args = args
        self.mode = mode
        self.num_negatives = num_negatives

        self.user_histories = []
        text_path = f"{args.data_path}/{args.data_name}.txt"

        if args.data_sample_K == 0:
            with open(text_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    self.user_histories.append((parts[0], parts[1:]))
        else:
            K = args.data_sample_K
            with open(text_path, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
            if mode == "train" and len(all_lines) > K:
                selected_lines = random.sample(all_lines, K)
            else:
                selected_lines = all_lines
            for line in selected_lines:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                self.user_histories.append((parts[0], parts[1:]))

        meta_path = f"{args.data_path}/{args.data_name}_item_meta.csv"
        meta = pd.read_csv(meta_path)
        meta["sid"] = meta["sid"].astype(str)

        self.id2title = dict(zip(meta["sid"], meta["title"]))
        self.all_items = list(self.id2title.keys())

        print(f"[{mode}] Loaded {len(self.user_histories)} users, {len(self.all_items)} items.")

        self.uids, self.histories, self.targets, self.neg_samples = [], [], [], []
        for user_id, seq in self.user_histories:
            if len(seq) < 3:
                continue
            self.uids.append(user_id)
            if mode == "train":
                self.histories.append(seq[:-2])
                self.targets.append(seq[-2])
            else:
                self.histories.append(seq[:-1])
                self.targets.append(seq[-1])
                pos_set = set(seq)
                neg_candidates = [i for i in self.all_items if i not in pos_set]
                sampled = random.sample(
                    neg_candidates, min(num_negatives, len(neg_candidates))
                )
                self.neg_samples.append(sampled)

    def make_history_string(self, item_ids):
        titles = [self.id2title[str(i)] for i in item_ids]
        numbered = [f"{idx}). {t}" for idx, t in enumerate(titles)]
        return numbered, ", ".join(numbered)

    def make_prompt(self, history_str, mode="input"):
        if mode == "input":
            return f"Describe this item title sequence in one word., {history_str}"
        return f"Describe this item title sequence in one word., 0). {history_str}"

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, idx):
        hist_ids = [int(x) for x in self.histories[idx]]
        title_list, hist_str = self.make_history_string(hist_ids)
        prompt = self.make_prompt(hist_str, mode="input")
        target_id = int(self.targets[idx])
        target_title = self.id2title[str(target_id)]

        if self.mode == "train":
            target_prompt = self.make_prompt(target_title, mode="target")
            return {
                "uid": self.uids[idx],
                "input_ids": hist_ids,
                "input_title_list": title_list,
                "history": prompt,
                "target_title": target_title,
                "target_ids": target_id,
                "target": target_prompt,
            }

        neg_ids = [int(nid) for nid in self.neg_samples[idx]]
        return {
            "input_ids": hist_ids,
            "history": prompt,
            "input_title_list": title_list,
            "target_ids": target_id,
            "target_title": target_title,
            "neg_indices": neg_ids,
        }

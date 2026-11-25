import random

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler


class RecDataset(Dataset):
    """Sequential recommendation dataset (leave-one-out style)."""

    def __init__(self, args, user_seq, mode="train"):
        self.args = args
        self.user_seq = []
        self.user_ids = []
        self.max_len = args.max_seq_length
        self.mode = mode

        if mode == "train":
            for user, seq in enumerate(user_seq):
                iids = seq[-(self.max_len + 2):-2]
                for i in range(len(iids)):
                    self.user_seq.append(iids[: i + 1])
                    self.user_ids.append(user)
        elif mode == "valid":
            for seq in user_seq:
                self.user_seq.append(seq[:-1])
        else:
            self.user_seq = user_seq

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, idx):
        seq = self.user_seq[idx]
        input_ids = seq[:-1]
        target = seq[-1]

        seq_set = set(seq)
        neg_target = neg_sample(seq_set, self.args.item_size)
        neg_candidates = neg_k_sample(seq_set, self.args.item_size, self.args.num_negatives)

        pad_len = self.max_len - len(input_ids)
        input_ids = ([0] * pad_len + input_ids)[-self.max_len:]
        assert len(input_ids) == self.max_len

        if self.mode in ("valid", "test"):
            return (
                torch.tensor(idx, dtype=torch.long),
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target, dtype=torch.long),
                torch.tensor(neg_candidates, dtype=torch.long),
            )

        return (
            torch.tensor(self.user_ids[idx], dtype=torch.long),
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(neg_target, dtype=torch.long),
        )


def neg_sample(seq_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in seq_set:
        item = random.randint(1, item_size - 1)
    return item


def neg_k_sample(seq_set, item_size, k):
    neg_items = []
    for _ in range(k):
        item = random.randint(1, item_size - 1)
        while item in seq_set:
            item = random.randint(1, item_size - 1)
        neg_items.append(item)
    return torch.tensor(neg_items, dtype=torch.long)


def get_seq_dic(args):
    args.data_file = args.data_dir + args.data_name + ".txt"
    user_seq = []
    item_set = set()
    with open(args.data_file, "r") as f:
        for line in f:
            _, seq = line.strip().split(" ", 1)
            seq = [int(x) for x in seq.split(" ")]
            user_seq.append(seq)
            item_set |= set(seq)
    return {
        "user_seq": user_seq,
        "num_users": len(user_seq),
        "num_items": max(item_set),
    }


def get_dataloader(args, seq_dic):
    train_dataset = RecDataset(args, seq_dic["user_seq"], mode="train")
    eval_dataset = RecDataset(args, seq_dic["user_seq"], mode="valid")
    test_dataset = RecDataset(args, seq_dic["user_seq"], mode="test")

    train_loader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size
    )
    eval_loader = DataLoader(
        eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=args.batch_size
    )
    test_loader = DataLoader(
        test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.batch_size
    )
    return train_loader, eval_loader, test_loader


def _generate_rating_matrix(user_seq, num_users, num_items, drop_last):
    row, col, data = [], [], []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-drop_last]:
            row.append(user_id)
            col.append(item)
            data.append(1)
    return csr_matrix(
        (np.array(data), (np.array(row), np.array(col))),
        shape=(num_users, num_items),
    )


def get_rating_matrix(seq_dic, max_item):
    num_items = max_item + 1
    valid = _generate_rating_matrix(
        seq_dic["user_seq"], seq_dic["num_users"], num_items, drop_last=2
    )
    test = _generate_rating_matrix(
        seq_dic["user_seq"], seq_dic["num_users"], num_items, drop_last=1
    )
    return valid, test

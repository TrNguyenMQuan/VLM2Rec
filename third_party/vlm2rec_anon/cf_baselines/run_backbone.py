"""Convenience launcher for downstream sequential recommendation backbones
trained on top of pretrained VLM2Rec item embeddings.

Currently shipped backbones: ``sasrec``. Other backbones (e.g. ``fmlp``,
``gru4rec``, ``diffurec``) can be plugged in by registering them in
``model/__init__.py`` and (optionally) adding their hyper-parameter knobs to
``MODEL_SPECIFIC_SEARCH`` below.

Generates a small bash script that calls ``main.py`` with each
hyper-parameter combination and executes it.
"""

import os
from dataclasses import asdict, dataclass
from itertools import product

from utils import get_local_time


# Per-backbone extra sweep entries. Merged into the base search space at script
# generation time so each model receives the hyper-parameters it actually consumes.
MODEL_SPECIFIC_SEARCH = {
    "sasrec": {},
}


search_space = {
    "model_name":  ["sasrec"],           # backbone(s); see MODEL_DICT in model/__init__.py
    "emb_model":   ["vlm2rec"],          # name of subdirectory holding fused .pt embeddings
    "data_name":   ["ml-1m"],            # ml-1m, Beauty, Toys_and_Games, Sports, Clothing
    "do_eval":     [False],
    "lm_embed":    [True],
    "lr":          [0.001],
    "hidden_size": [128],
    "emb_fusion":  ["concat"],           # "add" | "concat" | "modality_only"
    "act_fn":      ["gelu"],
    "gpu_id":      [0],
    "num_negatives": [0],
    "loss":        ["ce"],
    "epoch":       [300],
    "seed":        [42],
}


@dataclass
class Params:
    output_dir: str = "./output/"
    data_dir: str = "../datasets/processed/"
    item_emb_root: str = "../datasets/item_emb/"  # <root>/<data_name>/<emb_model>/*.pt

    batch_size: int = 256
    inner_size: int = 256
    max_seq_length: int = 50
    num_attention_heads: int = 2
    num_hidden_layers: int = 2
    hidden_dropout_prob: float = 0.5
    attn_dropout_prob: float = 0.5

    epochs: int = 300
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.0
    do_eval: bool = False
    seed: int = 42
    patience: int = 10
    log_freq: int = 1


def get_fixed_params():
    return asdict(Params())


def _expand_per_model(base_space):
    """Yield (model_name, expanded_space) — one expanded sweep per requested backbone."""
    model_names = base_space["model_name"]
    rest = {k: v for k, v in base_space.items() if k != "model_name"}
    for m in model_names:
        space = {"model_name": [m], **rest}
        space.update(MODEL_SPECIFIC_SEARCH.get(m, {}))
        yield m, space


def generate_script(shared_args, base_space):
    lines = ["#!/bin/bash", ""]
    shared = " ".join(f"--{k} {v}" for k, v in shared_args.items())

    for _, space in _expand_per_model(base_space):
        keys, value_lists = zip(*space.items())
        for values in product(*value_lists):
            combo = dict(zip(keys, values))
            args_line = " ".join(f"--{k} {v}" for k, v in combo.items())
            lines.append(f"python main.py {shared} {args_line}")
            lines.append("echo 'Finished one config.'")
            lines.append("sleep 2")
    return "\n".join(lines)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(search_space["gpu_id"][0])
    os.makedirs("search_scripts", exist_ok=True)

    shared = get_fixed_params()
    data_name = search_space["data_name"][0]
    model_name = "_".join(search_space["model_name"])
    emb_model = "_".join(search_space["emb_model"])

    script = generate_script(shared, search_space)
    path = f"search_scripts/{emb_model}_{model_name}_{data_name}_{get_local_time()}.sh"
    with open(path, "w") as f:
        f.write(script)

    print(f"[+] Created script: {path}")
    os.system(f"sh {path}")

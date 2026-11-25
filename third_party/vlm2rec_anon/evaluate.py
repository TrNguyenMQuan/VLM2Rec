import argparse
import csv
import glob
import os
import random
import re

import numpy as np
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from dataset import SRDataset
from metrics import evaluate
from models import VLEmbedder
from utils import eval_collate_fn, generate_item_embeddings, str2bool


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model(args, embedder, processor):
    args.image_prompt = "Describe this item image sequence in one word., "
    test_dataset = SRDataset(args, mode="test", num_negatives=args.num_neg_sample)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda b: eval_collate_fn(
            args, b, processor, args.img_dir, args.image_prompt
        ),
    )

    ks = [1, 5, 10] if args.num_neg_sample == 20 else [1, 5, 10, 20]
    results = evaluate(args, embedder, test_loader, processor, ks=ks)

    print("\n[Test Results]")
    for k in ks:
        print(f"Hit@{k}: {results[f'Hit@{k}']:.4f}, NDCG@{k}: {results[f'NDCG@{k}']:.4f}")

    csv_path = f"./results/evaluation/{args.data_name}/{args.load_desc}_eval.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "Hit@k", "NDCG@k"])
        for k in ks:
            writer.writerow([k, f"{results[f'Hit@{k}']:.4f}", f"{results[f'NDCG@{k}']:.4f}"])
    print(f"[+] Evaluation saved at {csv_path}")


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.load_desc = f"{args.loss_type}_{args.data_name}_{args.desc}_{args.id_time}"
    args.img_dir = f"{args.data_path}/{args.data_name}_image_pool"
    print(f"[*] Load Description: {args.load_desc}")

    if args.adapter_path is None:
        args.adapter_path = (
            f"{args.save_root}/{args.data_name}/{args.save_desc}/"
            f"{args.model_name.replace('/', '_')}/seed_{args.seed}/"
            f"epoch_{args.epochs}/cf_temp_{args.temp_cf}/aux_temp_{args.temp_aux}/"
            f"lmd_{args.lmd}/hf_model"
        )

    assert os.path.exists(args.adapter_path), (
        f"Adapter path does not exist: {args.adapter_path}"
    )

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    embedder = VLEmbedder(
        args.model_name,
        pooling_type=args.pooling_type,
        use_lora=False,
        padding_side="right",
        target_modules="auto",
        include_xmodal=False,
        use_gc=False,
        cache_dir=args.vlm_cache_dir,
        from_pretrained_kwargs={
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        },
    )
    embedder.to(args.device)
    embedder.eval()
    embedder.model = PeftModel.from_pretrained(embedder.model, args.adapter_path)
    print(f"[+] Loaded adapter from {args.adapter_path}")

    if args.phase in ["embed", "both"]:
        args.item_embeddings = generate_item_embeddings(args, embedder, processor)
    else:
        emb_dir = (
            f"./results/vlm_item_emb/{args.data_name}/{args.load_desc}/"
            f"epoch_{args.epochs}/{args.model_name.replace('/', '_')}/"
        )
        pt_files = sorted(glob.glob(os.path.join(emb_dir, "*.pt")))

        def extract_idx(path):
            return int(re.findall(r"\d+", os.path.basename(path))[-1])

        pt_files = sorted(pt_files, key=extract_idx)
        args.item_embeddings = torch.cat(
            [torch.load(f)["fused_embeddings"] for f in pt_files], dim=0
        )

    if args.phase in ["eval", "both"]:
        evaluate_model(args, embedder, processor)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VLM2Rec encoder")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data_name", type=str, default="ml-1m")
    parser.add_argument("--data_path", type=str, default="./data/processed")
    parser.add_argument("--vlm_cache_dir", type=str, default="./pretrained_vlm")
    parser.add_argument("--save_root", type=str, default="./checkpoints/")

    parser.add_argument("--loss_type", type=str, default="wpcl")
    parser.add_argument("--aux_loss", type=str, default="crtr")
    parser.add_argument("--pooling_type", type=str, default="last")
    parser.add_argument("--desc", type=str, default=".")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--temp_cf", type=float, default=0.05)
    parser.add_argument("--temp_aux", type=float, default=0.1)
    parser.add_argument("--lmd", type=float, default=0.1)

    parser.add_argument("--num_neg_sample", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--data_sample_K", type=int, default=0)
    parser.add_argument("--adapter_path", type=str, default=None)

    parser.add_argument("--phase", type=str, default="both", choices=["embed", "eval", "both"])

    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id_time", type=str, default="001")

    args = parser.parse_args()
    args.save_desc = f"{args.loss_type}_{args.data_name}_{args.desc}_{args.id_time}"
    return args


if __name__ == "__main__":
    args = parse_args()
    print("\n" + "=" * 60)
    print("EVALUATION CONFIGURATION")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"{k:<20}: {v}")
    print("=" * 60 + "\n")
    setup_seed(args.seed)
    main(args)

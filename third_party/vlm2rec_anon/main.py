import argparse
import os

import torch

from evaluate import main as eval_main
from train import Trainer
from utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser(description="Train + Evaluate VLM2Rec")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data_name", type=str, default="ml-1m")
    parser.add_argument("--data_path", type=str, default="./data/processed")
    parser.add_argument("--vlm_cache_dir", type=str, default="./pretrained_vlm")
    parser.add_argument("--save_root", type=str, default="./checkpoints/")

    parser.add_argument("--loss_type", type=str, default="wpcl", choices=["wpcl"])
    parser.add_argument("--aux_loss", type=str, default="crtr", choices=["crtr", "none"])
    parser.add_argument("--pooling_type", type=str, default="last", choices=["last", "mean"])
    parser.add_argument("--desc", type=str, default=".")

    parser.add_argument("--temp_cf", type=float, default=0.05)
    parser.add_argument("--temp_aux", type=float, default=0.1)
    parser.add_argument("--lmd", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--use_lora", type=str2bool, default=True)
    parser.add_argument("--use_gc", type=str2bool, default=True)
    parser.add_argument("--num_neg_sample", type=int, default=100)
    parser.add_argument("--data_sample_K", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--id_time", type=str, default="001")

    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--phase", type=str, default="both", choices=["embed", "eval", "both"])
    parser.add_argument("--adapter_path", type=str, default=None)

    args = parser.parse_args()
    args.save_desc = f"{args.loss_type}_{args.data_name}_{args.desc}_{args.id_time}"
    args.load_desc = args.save_desc
    args.img_dir = f"{args.data_path}/{args.data_name}_image_pool"
    args.image_prompt = "Describe this item image sequence in one word., "
    return args


def run_pipeline(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("\n[1/2] Training Phase...")
    trainer = Trainer(args)
    trainer.train()

    print("\n[2/2] Evaluation Phase...")
    torch.cuda.empty_cache()
    eval_main(args)

    print("\nPipeline complete. Results in ./results/evaluation\n")


if __name__ == "__main__":
    args = parse_args()
    print("\n" + "=" * 60)
    print("RUN CONFIGURATION")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"{k:<20}: {v}")
    print("=" * 60 + "\n")
    run_pipeline(args)

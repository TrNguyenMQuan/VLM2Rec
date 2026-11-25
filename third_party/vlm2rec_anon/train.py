import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from dataset import SRDataset
from loss import WPCL, CRTR
from models import VLEmbedder
from utils import collate_fn, process_inputs_train, str2bool


class Trainer:
    def __init__(self, args):
        self.args = args
        self._setup_seed(args.seed)

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args.epsilon = 1e-12
        self.args.model_family = "qwen"
        print(f"[*] Save Description: {args.save_desc}")

    @staticmethod
    def _setup_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train(self):
        processor = AutoProcessor.from_pretrained(
            self.args.model_name, use_fast=True, trust_remote_code=True
        )

        train_dataset = SRDataset(self.args, mode="train")
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(
                self.args, b, processor, self.args.img_dir, self.args.image_prompt
            ),
        )
        print(f"train_loader length : {len(train_loader)}")

        model = VLEmbedder(
            self.args.model_name,
            pooling_type=self.args.pooling_type,
            use_lora=self.args.use_lora,
            padding_side="right",
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.2,
            target_modules="auto",
            include_xmodal=False,
            use_gc=self.args.use_gc,
            cache_dir=self.args.vlm_cache_dir,
            from_pretrained_kwargs={
                "torch_dtype": torch.bfloat16,
                "trust_remote_code": True,
            },
        )

        # Modality-fusion weights are trained jointly with the encoder
        self.w_text = nn.Parameter(torch.tensor(1.0, device=self.args.device))
        self.w_image = nn.Parameter(torch.tensor(1.0, device=self.args.device))

        loss_fn = WPCL(self.args)
        loss_aux = CRTR(self.args) if self.args.aux_loss == "crtr" else None
        aux_loss_params = list(loss_aux.parameters()) if loss_aux is not None else []

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad]
            + list(loss_fn.parameters())
            + aux_loss_params
            + [self.w_text, self.w_image],
            lr=self.args.lr,
        )

        for epoch in range(1, self.args.epochs + 1):
            model.train()
            total_loss = 0.0
            pbar = tqdm(train_loader, desc=f"[Epoch {epoch}]")

            for batch in pbar:
                optimizer.zero_grad()
                input_images, input_texts, target_images, target_texts = process_inputs_train(
                    self.args, batch, processor
                )
                input_images = {k: v.to(self.args.device) for k, v in input_images.items()}
                input_texts = {k: v.to(self.args.device) for k, v in input_texts.items()}
                target_images = {k: v.to(self.args.device) for k, v in target_images.items()}
                target_texts = {k: v.to(self.args.device) for k, v in target_texts.items()}

                text_seq_repr = model(input_texts)
                text_target_repr = model(target_texts)
                image_seq_repr = model(input_images)
                image_target_repr = model(target_images)

                item_repr = (
                    text_seq_repr,
                    text_target_repr,
                    image_seq_repr,
                    image_target_repr,
                )

                fused_seq_repr = text_seq_repr + image_seq_repr
                fused_target_repr = text_target_repr + image_target_repr

                loss = loss_fn(fused_seq_repr, fused_target_repr, item_repr)
                if loss_aux is not None:
                    loss = loss + self.args.lmd * loss_aux(item_repr)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                torch.cuda.empty_cache()

            print(f"[*] Epoch {epoch} Avg Loss: {total_loss / len(train_loader):.4f}")

            if epoch in {1, self.args.epochs}:
                save_dir = os.path.join(
                    self.args.save_root,
                    f"{self.args.data_name}/{self.args.save_desc}/"
                    f"{self.args.model_name.replace('/', '_')}/seed_{self.args.seed}/"
                    f"epoch_{epoch}/cf_temp_{self.args.temp_cf}/aux_temp_{self.args.temp_aux}/"
                    f"lmd_{self.args.lmd}",
                )
                os.makedirs(save_dir, exist_ok=True)
                model.model.save_pretrained(os.path.join(save_dir, "hf_model"))
                processor.save_pretrained(os.path.join(save_dir, "hf_processor"))
                print(f"[+] Saved to: {save_dir}")

            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Train VLM2Rec encoder")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data_name", type=str, default="ml-1m")
    parser.add_argument("--data_path", type=str, default="./data/processed")
    parser.add_argument("--vlm_cache_dir", type=str, default="./pretrained_vlm")
    parser.add_argument("--save_root", type=str, default="./checkpoints/")

    parser.add_argument("--loss_type", type=str, default="wpcl", choices=["wpcl"])
    parser.add_argument("--aux_loss", type=str, default="crtr", choices=["crtr", "none"])
    parser.add_argument("--pooling_type", type=str, default="last", choices=["last", "mean"])
    parser.add_argument("--desc", type=str, default=".")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temp_cf", type=float, default=0.05)
    parser.add_argument("--temp_aux", type=float, default=0.1)
    parser.add_argument("--lmd", type=float, default=0.1)
    parser.add_argument("--use_lora", type=str2bool, default=True)
    parser.add_argument("--use_gc", type=str2bool, default=True)
    parser.add_argument("--num_neg_sample", type=int, default=100)
    parser.add_argument("--data_sample_K", type=int, default=0)

    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id_time", type=str, default="001")

    args = parser.parse_args()
    args.save_desc = f"{args.loss_type}_{args.data_name}_{args.desc}_{args.id_time}"
    args.img_dir = f"{args.data_path}/{args.data_name}_image_pool"
    args.image_prompt = "Describe this item image sequence in one word.,"
    return args


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()

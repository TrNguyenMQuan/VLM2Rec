import os
import re
import math
import glob
import argparse
from typing import Any, Dict

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from qwen_vl_utils import process_vision_info


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def load_image_from_index(index, image_pool_dir: str, size=(112, 112)) -> Image.Image:
    img_path = os.path.join(image_pool_dir, f"{index}.jpg")
    img = Image.open(img_path).convert("RGB")
    img = img.resize(size, Image.BICUBIC)
    return img


def _build_messages(item, image_pool_dir, image_prompt):
    imgs = [load_image_from_index(str(idx), image_pool_dir) for idx in item["input_ids"]]
    content = []
    for idx, img in enumerate(imgs):
        content.append({"type": "text", "text": f"{idx})"})
        content.append({"type": "image", "image": img})
    content.insert(0, {"type": "text", "text": image_prompt})
    return [{"role": "user", "content": content}]


def _build_target_messages(target_idx, image_pool_dir, image_prompt):
    target_img = load_image_from_index(str(target_idx), image_pool_dir)
    content = [
        {"type": "text", "text": image_prompt},
        {"type": "text", "text": "0)"},
        {"type": "image", "image": target_img},
    ]
    return [{"role": "user", "content": content}]


def collate_fn(args, batch, processor, image_pool_dir, image_prompt):
    """Collate items for VLM2Rec training (Qwen2.5-VL family)."""
    prefix_texts = [b["history"] for b in batch]
    target_titles = [b["target"] for b in batch]
    uids = [b["uid"] for b in batch]

    input_messages_list = [_build_messages(b, image_pool_dir, image_prompt) for b in batch]
    target_messages_list = [
        _build_target_messages(b["target_ids"], image_pool_dir, image_prompt) for b in batch
    ]

    chat_texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in input_messages_list
    ]
    target_chat_texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in target_messages_list
    ]

    image_inputs = []
    for m in input_messages_list:
        imgs, _ = process_vision_info(m, image_patch_size=14)
        image_inputs.append(imgs)

    target_image_inputs = []
    for m in target_messages_list:
        imgs, _ = process_vision_info(m, image_patch_size=14)
        target_image_inputs.append(imgs)

    return {
        "uids": uids,
        "chat_texts": chat_texts,
        "target_chat_texts": target_chat_texts,
        "image_inputs": image_inputs,
        "target_title": target_titles,
        "target_img": target_image_inputs,
        "input_titles": prefix_texts,
    }


def eval_collate_fn(args, batch, processor, image_pool_dir, image_prompt):
    """Collate items for evaluation."""
    prefix_texts = [b["history"] for b in batch]

    input_messages_list = [_build_messages(b, image_pool_dir, image_prompt) for b in batch]
    chat_texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in input_messages_list
    ]
    image_inputs = []
    for m in input_messages_list:
        imgs, _ = process_vision_info(m, image_patch_size=14)
        image_inputs.append(imgs)

    return {
        "chat_texts": chat_texts,
        "image_inputs": image_inputs,
        "input_titles": prefix_texts,
        "target_ids": [b["target_ids"] for b in batch],
        "neg_indices": [b["neg_indices"] for b in batch],
    }


def process_inputs(input_text, input_image, processor, modality, max_len: int = 1000):
    if modality == "text":
        return processor(
            text=input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
    return processor(
        text=input_text,
        images=input_image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )


def process_inputs_train(args, batch: Dict[str, Any], processor, max_text_len: int = 128):
    input_texts = process_inputs(batch["input_titles"], None, processor, modality="text")
    target_texts = process_inputs(batch["target_title"], None, processor, modality="text")

    input_images = process_inputs(
        batch["chat_texts"], batch["image_inputs"], processor, modality="input_image"
    ).to(dtype=torch.bfloat16)
    target_images = process_inputs(
        batch["target_chat_texts"], batch["target_img"], processor, modality="target_image"
    ).to(dtype=torch.bfloat16)

    target_texts["labels"] = target_texts["input_ids"].clone()
    target_images["labels"] = target_images["input_ids"].clone()
    return input_images, input_texts, target_images, target_texts


def process_inputs_eval(batch: Dict[str, Any], processor):
    input_texts = process_inputs(
        batch["input_titles"], None, processor, modality="text"
    ).to(dtype=torch.bfloat16)
    input_images = process_inputs(
        batch["chat_texts"], batch["image_inputs"], processor, modality="input_image"
    ).to(dtype=torch.bfloat16)
    return input_images, input_texts


def generate_item_embeddings(args, embedder, processor, fusion_weights=None):
    """Encode every catalog item with the VLM and cache fused/text/image embeddings."""
    df = pd.read_csv(f"{args.data_path}/{args.data_name}_item_meta.csv")
    image_pool_dir = f"{args.data_path}/{args.data_name}_image_pool/"
    save_dir = (
        f"./results/vlm_item_emb/{args.data_name}/{args.load_desc}"
        f"/epoch_{args.epochs}/{args.model_name.replace('/', '_')}/"
    )
    os.makedirs(save_dir, exist_ok=True)

    existing_pt_files = sorted(glob.glob(os.path.join(save_dir, "item_emb_*.pt")))

    def extract_idx(path):
        fname = os.path.basename(path)
        num = re.findall(r"\d+", fname)
        return int(num[-1])

    existing_pt_files = sorted(existing_pt_files, key=extract_idx)
    if len(existing_pt_files) == math.ceil(len(df) / args.batch_size):
        print(f"Found existing item embeddings in {save_dir}. Skipping generation.")
        return torch.cat(
            [torch.load(f)["fused_embeddings"] for f in existing_pt_files], dim=0
        )

    image_prompt = "Describe this item image sequence in one word.,"
    batch_size = args.batch_size

    input_messages_list = []
    image_inputs = []
    print("Preparing image-text inputs...")
    for idx in range(1, len(df) + 1):
        img = load_image_from_index(idx, image_pool_dir)
        content = [
            {"type": "text", "text": image_prompt},
            {"type": "text", "text": f"{idx})"},
            {"type": "image", "image": img},
        ]
        input_messages_list.append([{"role": "user", "content": content}])

    chat_texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in input_messages_list
    ]
    for m in input_messages_list:
        imgs, _ = process_vision_info(m, image_patch_size=14)
        image_inputs.append(imgs)

    print(f"Generating embeddings for {len(df)} items...")
    if fusion_weights is not None:
        w_text, w_image = fusion_weights

    all_fused = []
    for batch_idx, start in enumerate(tqdm(range(0, len(df), batch_size))):
        end = min(start + batch_size, len(df))
        batch_df = df.iloc[start:end]
        texts = [
            f"Describe this item title sequence in one word., 0){row['title']}"
            for _, row in batch_df.iterrows()
        ]

        text_input = processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(args.device).to(dtype=torch.bfloat16)

        image_input = processor(
            text=chat_texts[start:end],
            images=image_inputs[start:end],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(args.device).to(dtype=torch.bfloat16)

        with torch.no_grad():
            text_embeds = embedder(text_input)
            img_embeds = embedder(image_input)
            if fusion_weights is not None:
                fused_embeds = (w_text * text_embeds) + (w_image * img_embeds)
            else:
                fused_embeds = text_embeds + img_embeds
            fused_embeds = fused_embeds.cpu()

        all_fused.append(fused_embeds)
        sids = torch.tensor(batch_df["sid"].tolist())
        torch.save(
            {
                "sids": sids,
                "fused_embeddings": fused_embeds,
                "text_embeddings": text_embeds.cpu(),
                "image_embeddings": img_embeds.cpu(),
            },
            os.path.join(save_dir, f"item_emb_{batch_idx + 1}.pt"),
        )
        torch.cuda.empty_cache()

    print("Item embeddings generated.")
    return torch.cat(all_fused, dim=0)

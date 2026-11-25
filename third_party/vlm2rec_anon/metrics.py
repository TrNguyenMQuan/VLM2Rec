import numpy as np
import torch
from tqdm import tqdm

from utils import process_inputs_eval


def hit_at_k(preds, targets, k):
    hits = 0
    for pred, target in zip(preds, targets):
        if target in pred[:k]:
            hits += 1
    return hits / len(targets)


def ndcg_at_k(preds, targets, k):
    score = 0.0
    for pred, target in zip(preds, targets):
        pred_k = pred[:k]
        if isinstance(pred_k, np.ndarray):
            pred_k = pred_k.tolist()
        if target in pred_k:
            rank = pred_k.index(target)
            score += 1.0 / np.log2(rank + 2)
    return score / len(targets)


def evaluate(args, model, dataloader, processor, ks=(5, 10, 20)):
    """Score Hit@k / NDCG@k by ranking the ground-truth target against sampled negatives."""
    model.eval()
    device = args.device

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Eval]", total=len(dataloader)):
            input_images, input_texts = process_inputs_eval(batch, processor)
            input_images = {k: v.to(device) for k, v in input_images.items()}
            input_texts = {k: v.to(device) for k, v in input_texts.items()}

            text_seq_repr = model(input_texts)
            image_seq_repr = model(input_images)
            seq_repr = text_seq_repr + image_seq_repr

            target_repr = args.item_embeddings[
                [tid - 1 for tid in batch["target_ids"]]
            ].to(device)

            if args.num_neg_sample:
                batch_neg_repr = []
                for neg_idx_list in batch["neg_indices"]:
                    neg_embs = args.item_embeddings[
                        [nid - 1 for nid in neg_idx_list]
                    ].to(device)
                    batch_neg_repr.append(neg_embs)
                cand_repr = torch.stack(batch_neg_repr, dim=0)
                cand_repr = torch.cat([target_repr.unsqueeze(1), cand_repr], dim=1)
                similarities = torch.bmm(
                    seq_repr.unsqueeze(1), cand_repr.transpose(1, 2)
                ).squeeze(1)
            else:
                cand_repr = args.item_embeddings.to(device)
                similarities = torch.einsum("bd,nd->bn", seq_repr, cand_repr)

            _, ranked = torch.sort(similarities, dim=1, descending=True)
            preds = ranked.cpu().numpy()
            targets = torch.zeros(similarities.size(0), dtype=torch.long).numpy()

            all_preds.extend(preds)
            all_targets.extend(targets)
            torch.cuda.empty_cache()

    results = {}
    for k in ks:
        results[f"Hit@{k}"] = hit_at_k(all_preds, all_targets, k)
        results[f"NDCG@{k}"] = ndcg_at_k(all_preds, all_targets, k)
    return results

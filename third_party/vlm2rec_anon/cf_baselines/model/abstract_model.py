import glob
import os
import re

import torch
import torch.nn as nn


class Emb(nn.Module):
    """Item embedding layer.

    Supports two paths:
    - lm_embed=False: pure ID embedding.
    - lm_embed=True : ID embedding fused with frozen pretrained item embeddings
                      (`fused_embeddings` produced by VLM2Rec) via 'add' / 'concat' /
                      'modality_only'.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.lm_embed = bool(args.lm_embed)
        self.fusion_mode = "id_only"

        self.pretrained_item_embeddings = None
        self.projector = None

        if self.lm_embed:
            self._init_single_embedding_mode(args)
        else:
            self.item_id_embeddings = nn.Embedding(
                args.item_size, args.hidden_size, padding_idx=0
            )
            self.output_dim = args.hidden_size

        self._init_id_embedding()

    def _init_single_embedding_mode(self, args):
        pretrained = self._load_pretrained_embeddings(args.pretrained_item_emb_path)
        args.lm_dim = pretrained.shape[1]

        self.pretrained_item_embeddings = nn.Embedding.from_pretrained(
            embeddings=self._add_padding_row(pretrained),
            freeze=True,
            padding_idx=0,
        )
        self.pretrained_item_embeddings.weight.requires_grad = False

        emb_fusion = getattr(args, "emb_fusion", "add")
        if emb_fusion == "concat":
            self.fusion_mode = "concat"
            self.output_dim = args.hidden_size * 2
        elif emb_fusion == "modality_only":
            self.fusion_mode = "modality_only"
            self.output_dim = args.hidden_size
        else:
            self.fusion_mode = "add"
            self.output_dim = args.hidden_size

        self.item_id_embeddings = nn.Embedding(
            args.item_size, args.hidden_size, padding_idx=0
        )
        self.projector = nn.Linear(args.lm_dim, args.hidden_size, bias=False)

    @staticmethod
    def _load_pretrained_embeddings(folder_path):
        print(f"Loading pretrained item embeddings from {folder_path}...")
        pt_files = sorted(glob.glob(os.path.join(folder_path, "*_*.pt")))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {folder_path}")

        def natural_sort_key(filename):
            match = re.search(r"item_emb_(\d+)\.pt$", os.path.basename(filename))
            return int(match.group(1)) if match else float("inf")

        pt_files = sorted(pt_files, key=natural_sort_key)
        embeddings_list = []
        for pt_file in pt_files:
            emb = torch.load(pt_file)
            for candidate in ("fused_embeddings", "fused_emb", "embeddings"):
                if candidate in emb:
                    embeddings_list.append(emb[candidate])
                    break
            else:
                raise KeyError(
                    f"None of ('fused_embeddings', 'fused_emb', 'embeddings') found in {pt_file}. "
                    f"Available keys: {list(emb.keys())}"
                )
        return torch.cat(embeddings_list, dim=0).float()

    @staticmethod
    def _add_padding_row(embs):
        padding = torch.zeros(1, embs.shape[-1], dtype=embs.dtype, device=embs.device)
        return torch.cat([padding, embs], dim=0)

    def _init_id_embedding(self):
        self.item_id_embeddings.weight.data.normal_(
            mean=0.0, std=self.args.initializer_range
        )
        if self.item_id_embeddings.padding_idx is not None:
            with torch.no_grad():
                self.item_id_embeddings.weight[self.item_id_embeddings.padding_idx].zero_()

    def forward(self, sequence):
        id_emb = self.item_id_embeddings(sequence)
        if not self.lm_embed:
            return id_emb

        frozen = self.pretrained_item_embeddings(sequence)
        projected = self.projector(frozen)

        if self.fusion_mode == "modality_only":
            return projected
        if self.fusion_mode == "concat":
            return torch.cat([id_emb, projected], dim=-1)
        return id_emb + projected

    def get_item_embedding_weights(self):
        id_weight = self.item_id_embeddings.weight
        if not self.lm_embed:
            return id_weight

        projected_weight = self.projector(self.pretrained_item_embeddings.weight)
        if self.fusion_mode == "modality_only":
            return projected_weight
        if self.fusion_mode == "concat":
            return torch.cat([id_weight, projected_weight], dim=-1)
        return id_weight + projected_weight


class AbstractModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.item_embedding_layer = Emb(args)
        enc_dim = self.item_embedding_layer.output_dim

        self.position_embeddings = nn.Embedding(args.max_seq_length, enc_dim)
        self.position_embeddings.weight.data.normal_(
            mean=0.0, std=self.args.initializer_range
        )

        self.batch_size = args.batch_size
        self.LayerNorm = nn.LayerNorm(enc_dim, eps=args.layer_norm_eps)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def get_sequence_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)

        item_embeddings = self.item_embedding_layer(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_embeddings = item_embeddings + position_embeddings
        sequence_embeddings = self.dropout(self.LayerNorm(sequence_embeddings))
        return sequence_embeddings

    def get_item_embedding_weights(self):
        return self.item_embedding_layer.get_item_embedding_weights()

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq, bidirectional=False):
        attention_mask = item_seq != 0
        extended = attention_mask.unsqueeze(1).unsqueeze(2)
        if not bidirectional:
            extended = torch.tril(extended.expand((-1, -1, item_seq.size(-1), -1)))
        return torch.where(extended, 0.0, -10000.0)

    def forward(self, input_ids):
        raise NotImplementedError

    def predict(self, input_ids):
        return self.forward(input_ids)

    def calculate_loss(self, input_ids, answers, neg_answers):
        raise NotImplementedError

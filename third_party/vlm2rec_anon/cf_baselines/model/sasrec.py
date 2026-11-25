import torch
import torch.nn as nn

from model.abstract_model import AbstractModel
from model.sasrec_modules import TransformerEncoder


class SASRec(AbstractModel):
    """SASRec backbone consuming ID + (optional) frozen pretrained item embeddings."""

    def __init__(self, args):
        super().__init__(args)
        args.hidden_size = self.item_embedding_layer.output_dim
        self.item_encoder = TransformerEncoder(args)
        self.apply(self.init_weights)

    def forward(self, item_seq):
        extended_attention_mask = self.get_attention_mask(item_seq)
        sequence_embeddings = self.get_sequence_embedding(item_seq)
        sequence_output = self.item_encoder(sequence_embeddings, extended_attention_mask)
        return sequence_output

    def calculate_loss(self, item_seq, answers, neg_answers):
        sequence_output = self.forward(item_seq)
        sequence_embedding = sequence_output[:, -1, :]

        if self.args.loss == "bce":
            pos_emb = self.item_embedding_layer(answers)
            neg_emb = self.item_embedding_layer(neg_answers)

            pos_logits = torch.sum(sequence_embedding * pos_emb, dim=-1)
            neg_logits = torch.sum(sequence_embedding * neg_emb, dim=-1)

            pos_labels = torch.ones(pos_logits.shape, device=sequence_embedding.device)
            neg_labels = torch.zeros(neg_logits.shape, device=sequence_embedding.device)
            indices = (answers != 0).nonzero().reshape(-1)

            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(pos_logits[indices], pos_labels[indices])
            loss = loss + criterion(neg_logits[indices], neg_labels[indices])
            return loss

        # default: cross-entropy over the full item set
        test_item_emb = self.get_item_embedding_weights()
        logits = torch.matmul(sequence_embedding, test_item_emb.transpose(0, 1))
        return nn.CrossEntropyLoss()(logits, answers)

    def predict(self, item_seq):
        seq_out = self.forward(item_seq)
        return seq_out[:, -1, :]

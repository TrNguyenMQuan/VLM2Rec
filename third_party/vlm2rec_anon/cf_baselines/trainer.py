import numpy as np
import torch
import tqdm
from torch.optim import Adam

from metrics import ndcg_k, recall_at_k


class Trainer:
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args, logger):
        self.args = args
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        if torch.cuda.is_available():
            self.model.cuda()

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(
            self.model.parameters(),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        self.logger.info(
            f"Total Parameters: {sum(p.nelement() for p in self.model.parameters())}"
        )

    def train(self, epoch):
        self._iterate(epoch, self.train_dataloader, train=True)

    def valid(self, epoch):
        self.args.train_matrix = self.args.valid_rating_matrix
        return self._iterate(epoch, self.eval_dataloader, train=False)

    def test(self, epoch):
        self.args.train_matrix = self.args.test_rating_matrix
        return self._iterate(epoch, self.test_dataloader, train=False)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        state = torch.load(file_name)
        self.model.load_state_dict(state)

    def _predict_full(self, seq_out):
        item_emb = self.model.get_item_embedding_weights()
        return torch.matmul(seq_out, item_emb.transpose(0, 1))

    def _predict_sampled(self, seq_out, candidates):
        item_emb = self.model.get_item_embedding_weights()
        cand_emb = item_emb[candidates]
        return torch.bmm(seq_out.unsqueeze(1), cand_emb.transpose(1, 2)).squeeze(1)

    def _full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [1, 5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HR@5": f"{recall[1]:.4f}", "HR@10": f"{recall[2]:.4f}", "HR@20": f"{recall[4]:.4f}",
            "NDCG@5": f"{ndcg[1]:.4f}", "NDCG@10": f"{ndcg[2]:.4f}", "NDCG@20": f"{ndcg[4]:.4f}",
        }
        self.logger.info(post_fix)
        return (
            [recall[0], ndcg[0], recall[1], ndcg[1], recall[2], ndcg[2], recall[4], ndcg[4]],
            str(post_fix),
        )

    def _iterate(self, epoch, dataloader, train=True):
        bar = tqdm.tqdm(
            enumerate(dataloader),
            desc=f"Mode_{'train' if train else 'test'}:{epoch}",
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}",
        )

        if train:
            self.model.train()
            rec_loss = 0.0
            for _, batch in bar:
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, answers, neg_answers = batch
                loss = self.model.calculate_loss(input_ids, answers, neg_answers)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                rec_loss += loss.item()

            if (epoch + 1) % self.args.log_freq == 0:
                self.logger.info({
                    "epoch": epoch,
                    "rec_loss": f"{rec_loss / len(bar):.4f}",
                })
            return None

        self.model.eval()
        pred_list, answer_list = None, None
        with torch.no_grad():
            for i, batch in bar:
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, answers, neg_answers = batch
                seq_out = self.model.predict(input_ids)

                if int(self.args.num_negatives) > 0:
                    candidates = torch.cat([answers.unsqueeze(1), neg_answers], dim=1)
                    rating_pred = self._predict_sampled(seq_out, candidates)
                    rating_pred = rating_pred.cpu().data.numpy().copy()
                else:
                    rating_pred = self._predict_full(seq_out)
                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[
                        self.args.train_matrix[batch_user_index].toarray() > 0
                    ] = 0

                ind = np.argpartition(rating_pred, -20)[:, -20:]
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

                if int(self.args.num_negatives) > 0:
                    batch_pred_list = candidates[
                        np.arange(len(rating_pred))[:, None],
                        ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort],
                    ]
                    batch_pred_list = batch_pred_list.cpu().numpy()
                else:
                    batch_pred_list = ind[
                        np.arange(len(rating_pred))[:, None], arr_ind_argsort
                    ]

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)

        return self._full_sort_score(epoch, answer_list, pred_list)

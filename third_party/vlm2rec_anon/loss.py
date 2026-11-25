import torch
import torch.nn as nn
import torch.nn.functional as F


class WPCL(nn.Module):
    """Weak-modality Penalized Contrastive Loss (main objective)."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.epsilon = 1e-12
        self.beta = nn.Parameter(torch.FloatTensor([0.8]), requires_grad=True).to(args.device)
        self.tau = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True).to(args.device)
        self.ce = nn.CrossEntropyLoss()

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, fused_seq_repr, fused_target_repr, item_repr):
        text_seq_repr, text_target_repr, image_seq_repr, image_target_repr = item_repr

        B = fused_seq_repr.size(0)
        N = 2 * B

        z_f = torch.cat((fused_seq_repr, fused_target_repr), dim=0)
        z_t = torch.cat((text_seq_repr, text_target_repr), dim=0)
        z_i = torch.cat((image_seq_repr, image_target_repr), dim=0)

        sim_f = torch.mm(z_f, z_f.T) / self.args.temp_cf
        sim_t = torch.mm(z_t, z_t.T) / self.args.temp_cf
        sim_i = torch.mm(z_i, z_i.T) / self.args.temp_cf

        mask = self.mask_correlated_samples(B)

        sim_t_neg = sim_t[mask].reshape(N, -1)
        sim_i_neg = sim_i[mask].reshape(N, -1)
        pos_t = torch.cat((torch.diag(sim_t, B), torch.diag(sim_t, -B)), 0)
        pos_i = torch.cat((torch.diag(sim_i, B), torch.diag(sim_i, -B)), 0)

        margin_t = pos_t.unsqueeze(1) - sim_t_neg
        margin_i = pos_i.unsqueeze(1) - sim_i_neg

        is_text_weak = (margin_t < margin_i)
        m_weak = torch.where(is_text_weak, margin_t, margin_i)
        m_strong = torch.where(is_text_weak, margin_i, margin_t)
        gap = (m_strong.detach() - m_weak)
        w = 1.0 + torch.sigmoid(self.beta) * F.softplus(torch.sigmoid(self.tau) * gap)

        sim_i_j = torch.diag(sim_f, B)
        sim_j_i = torch.diag(sim_f, -B)
        pos_f = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        neg_f = sim_f[mask].reshape(N, -1)

        weighted_neg = neg_f + torch.log(w + self.epsilon)
        logits = torch.cat((pos_f, weighted_neg), dim=1)

        labels = torch.zeros(N, dtype=torch.long, device=logits.device)
        loss = self.ce(logits, labels)
        return loss


class CRTR(nn.Module):
    """Cross-modal Relational Topology Regularization (auxiliary loss)."""

    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, item_repr):
        text_seq_repr, text_target_repr, image_seq_repr, image_target_repr = item_repr

        sim_t = torch.matmul(text_seq_repr, text_target_repr.T)
        sim_i = torch.matmul(image_seq_repr, image_target_repr.T)

        P_t = F.softmax(sim_t, dim=-1) / self.args.temp_aux
        P_i = F.softmax(sim_i, dim=-1) / self.args.temp_aux

        kl_ti = (P_t * (torch.log(P_t + self.args.epsilon) - torch.log(P_i + self.args.epsilon))).sum(dim=-1)
        kl_it = (P_i * (torch.log(P_i + self.args.epsilon) - torch.log(P_t + self.args.epsilon))).sum(dim=-1)

        loss = 0.5 * (kl_ti + kl_it).mean()
        return loss

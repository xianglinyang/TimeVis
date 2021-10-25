import torch
from torch import nn
from umap.umap_ import find_ab_params

from utils import convert_distance_to_probability, compute_cross_entropy

"""Losses modules for preserving four propertes"""
# https://github.com/ynjnpa/VocGAN/blob/5339ee1d46b8337205bec5e921897de30a9211a1/utils/stft_loss.py

class UmapLoss(nn.Module):
    def __init__(self, negative_sample_rate, _a, _b, repulsion_strength=1.0):
        super(UmapLoss, self).__init__()

        self._negative_sample_rate = negative_sample_rate
        self._a = _a,
        self._b = _b,
        self._repulsion_strength = repulsion_strength

    def forward(self, embedding_to, embedding_from):
        batch_size = embedding_to.shape[0]
        # get negative samples
        embedding_neg_to = torch.repeat_interleave(embedding_to, self._negative_sample_rate, dim=0)
        repeat_neg = torch.repeat_interleave(embedding_from, self._negative_sample_rate, dim=0)
        randperm = torch.randperm(repeat_neg.shape[0])
        embedding_neg_from = repeat_neg[randperm]

        #  distances between samples (and negative samples)
        distance_embedding = torch.cat(
            (
                torch.norm(embedding_to - embedding_from, dim=1),
                torch.norm(embedding_neg_to - embedding_neg_from, dim=1),
            ),
            dim=0,
        )
        probabilities_distance = convert_distance_to_probability(
            distance_embedding, self._a, self._b
        )

        # set true probabilities based on negative sampling
        probabilities_graph = torch.cat(
            (torch.ones(batch_size), torch.zeros(batch_size * self._negative_sample_rate)), dim=0,
        )
        # true probabilies in high-dimensional space
        # probabilities = torch.cat(
        #     (torch.squeeze(weights), torch.zeros(batch_size * self._negative_sample_rate)), dim=0,
        # )

        # compute cross entropy
        (_, _, ce_loss) = compute_cross_entropy(
            probabilities_graph,
            probabilities_distance,
            repulsion_strength=self._repulsion_strength,
        )

        return torch.mean(ce_loss)


class ReconstructionLoss(nn.Module):
    def __init__(self, beta=1.0):
        super(ReconstructionLoss, self).__init__()
        self._beta = beta

    def forward(self, edge_to, edge_from, recon_to, recon_from, alpha_to, alpha_from):
        loss1 = torch.mean(torch.mean(torch.multiply(torch.pow((1+alpha_to), self._beta), torch.pow(edge_to - recon_to, 2)), 1))
        loss2 = torch.mean(torch.mean(torch.multiply(torch.pow((1+alpha_from), self._beta), torch.pow(edge_from - recon_from, 2)), 1))
        return (loss1 + loss2)/2


# class SingleVisLoss(nn.Module):
#     def __init__(self):
#         super(SingleVisLoss, self).__init__()
#         self.umap_loss = UmapLoss()
#         self.recon_loss = ReconstructionLoss()
#     def forward(self,):

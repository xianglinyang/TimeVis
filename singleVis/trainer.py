import torch
import os
import numpy as np
from evaluate import evaluate_proj_nn_perseverance_knn
"""
1. construct a spatio-temporal complex
2. construct an edge-dataset
3. train the network

Trainer should contains
1. train_step function
2. early stop
3....
"""


class SingleVisTrainer:
    def __init__(self, model, criterion, optimizer, edge_loader, DEVICE):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.DEVICE = DEVICE
        self.edge_loader = edge_loader
        self._loss = 100.0

    @property
    def loss(self):
        return self._loss

    def train_step(self):
        self.model.to(device=self.DEVICE)
        self.model.train()
        all_loss = []
        umap_losses = []
        recon_losses = []
        for data in self.edge_loader:
            edge_to, edge_from = data

            edge_to = edge_to.to(device=self.DEVICE, dtype=torch.float32)
            edge_from = edge_from.to(device=self.DEVICE, dtype=torch.float32)

            outputs = self.model(edge_to, edge_from)
            umap_l, recon_l, loss = self.criterion(edge_to, edge_from, outputs)
            all_loss.append(loss.item())
            umap_losses.append(umap_l.item())
            recon_losses.append(recon_l.item())
            # ===================backward====================
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self._loss = sum(all_loss) / len(all_loss)
        self.model.eval()
        print('umap:{:.4f}\trecon_l:{:.4f}\tloss:{:.4f}'.format(sum(umap_losses) / len(umap_losses),
                                                                sum(recon_losses) / len(recon_losses),
                                                                sum(all_loss) / len(all_loss)))
        return self.loss

    def load(self, device, name="singleVisModel"):
        """
        save all parameters...
        :param name:
        :return:
        """
        save_model = torch.load(name + '.pth')
        self._loss = save_model["loss"]
        self.model.load_state_dict(save_model["state_dict"])
        self.optimizer.load_state_dict(save_model["optimizer"])
        self.model.to(device)
        self.optimizer.to(device)
        print("Successfully load visualization mdoel")

    def save(self, name="singleVisModel"):
        """
        save all parameters...
        :param name:
        :return:
        """
        save_model = {
            "loss": self.loss,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()}
        torch.save(save_model, name + '.pth')
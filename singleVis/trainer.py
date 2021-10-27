import torch
from torch.utils.data import DataLoader

import edge_dataset
from SingleVisualizationModel import SingleVisualizationModel
from losses import SingleVisLoss
from edge_dataset import DataHandler
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
        self.model.train()
        all_loss = []
        for data in self.edge_loader:
            edge_to, edge_from = data

            edge_to.to(device=self.DEVICE, dtype=torch.float32)
            edge_from.to(device=self.DEVICE, dtype=torch.float32)

            outputs = self.model(edge_to, edge_from)
            loss = self.criterion(outputs)
            all_loss.append(loss.item())
            # ===================backward====================
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self._loss = sum(all_loss) / len(all_loss)
        self.model.eval()
        print('loss:{:.4f}'.format(sum(all_loss) / len(all_loss)))
        return self.loss

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
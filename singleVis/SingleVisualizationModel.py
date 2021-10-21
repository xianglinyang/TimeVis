import os
import torch
from torch import nn


class SingleVisualizationModel(nn.Module):
    def __init__(self, input_dims, output_dims, units):
        super(SingleVisualizationModel, self).__init__()

        # TODO find the best model architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, units),
            nn.ReLU(True),
            nn.Linear(units, units),
            nn.ReLU(True),
            nn.Linear(units, units),
            nn.ReLU(True),
            nn.Linear(units, units),
            nn.ReLU(True),
            nn.Linear(units, output_dims)
        )

        self.decoder = nn.Sequential(
            nn.Linear(output_dims, units),
            nn.ReLU(True),
            nn.Linear(units, units),
            nn.ReLU(True),
            nn.Linear(units, units),
            nn.ReLU(True),
            nn.Linear(units, units),
            nn.ReLU(True),
            nn.Linear(units, input_dims)
        )

    def forward(self, x):
        # defind model function
        return x
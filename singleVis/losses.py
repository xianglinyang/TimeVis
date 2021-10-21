from torch import nn

"""Losses modules for preserving four propertes"""
# https://github.com/ynjnpa/VocGAN/blob/5339ee1d46b8337205bec5e921897de30a9211a1/utils/stft_loss.py

class UmapLoss(nn.Module):
    def __init__(self):
        super(UmapLoss, self).__init__()

    # def forward(self):


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()


class SingleVisLoss(nn.Module):
    def __init__(self):
        super(SingleVisLoss, self).__init__()
    # init multiple losses modules
    # def forward(self,):
import torch
import sys
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from SingleVisualizationModel import SingleVisualizationModel
from losses import SingleVisLoss, UmapLoss, ReconstructionLoss
from edge_dataset import DataHandler
from trainer import SingleVisTrainer
from backend import fuzzy_complex, boundary_wise_complex, construct_step_edge_dataset


# define parameters
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCH_NUMS = 100
LEN = 50000
TIME_STEP = 7
TEMPORAL_PERSISTANT = 2



model = SingleVisualizationModel()
criterion = SingleVisLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=.1)

content_path = "E:\\DVI_exp_data\\resnet18_cifar10"
sys.path.append(content_path)
from Model.model import *
net = resnet18()
classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
selected_idxs = np.random.choice(np.arange(LEN), size=LEN//10, replace=False)

# TODO construct spatio-temporal complex and get edges
# dummy input
edge_to = None
edge_from = None
weight = None
feature_vectors = None


for t in range(TIME_STEP):
    train_data_loc = os.path.join(os.path.join(content_path, "Model"), "Epoch_{:d}".format(t), "train_data.npy")
    try:
        train_data = np.load(train_data_loc)
        train_data = train_data[selected_idxs]
    except Exception as e:
        print("no train data saved for Epoch {}".format(t))
        continue
    border_centers_loc = os.path.join(os.path.join(content_path, "Model"), "Epoch_{:d}".format(t),
                                                   "advance_border_centers.npy")
    try:
        border_centers = np.load(border_centers_loc)
    except Exception as e:
        print("no border points saved for Epoch {}".format(t))
        continue

    complex, sigmas, rhos = fuzzy_complex(train_data, 15)
    bw_complex, _, _ = boundary_wise_complex(train_data, border_centers, 15)
    edge_to_t, edge_from_t, weight_t = construct_step_edge_dataset((train_data, border_centers), complex, bw_complex)
    fitting_data = np.concatenate((train_data, border_centers), axis=0)
    if edge_to is None:
        edge_to = edge_to_t[None, :]
        edge_from = edge_from_t[None, :]
        weight = weight_t[None, :]
        feature_vectors = fitting_data[None, :]
    else:
        edge_to = np.concatenate((edge_to, edge_to_t[None, :]), axis=0)
        edge_from = np.concatenate((edge_from, edge_from_t[None, :]), axis=0)
        weight = np.concatenate((weight, weight_t[None, :]), axis=0)
        feature_vectors = np.concatenate((feature_vectors, fitting_data[None, :]), axis=0)
dataset = DataHandler(edge_to, edge_from, feature_vectors)
probs = weight



# TODO NUMS decided by make_epoch_per_sample
sampler = WeightedRandomSampler(probs, NUMS, replacement=False)
edge_loader = DataLoader(dataset, batch_size=1000, sampler=sampler)


trainer = SingleVisTrainer(model, criterion, optimizer, edge_loader=edge_loader, DEVICE=DEVICE)
for epoch in EPOCH_NUMS:
    trainer.train_step()
    # early stop, check whether converge or not
trainer.save(name="cifar10_epoch_10")

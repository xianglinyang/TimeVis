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

from backend import fuzzy_complex, boundary_wise_complex, construct_step_edge_dataset, construct_temporal_edge_dataset


# define hyperparameters
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCH_NUMS = 100
LEN = 50000
TIME_STEPS = 7
TEMPORAL_PERSISTANT = 2
NUMS = 5    # how many epoch should we go through for one pass
PATIENT = 4

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
sigmas = None
rhos = None
weight = None
probs = None
feature_vectors = None
knn_indices = None
n_vertices = -1

# each time step

for t in range(TIME_STEPS):
    # load train data and border centers
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

    complex, sigmas_t1, rhos_t1, knn_idxs_t = fuzzy_complex(train_data, 15)
    bw_complex, sigmas_t2, rhos_t2, _ = boundary_wise_complex(train_data, border_centers, 15)
    edge_to_t, edge_from_t, weight_t = construct_step_edge_dataset((train_data, border_centers), complex, bw_complex)
    sigmas_t = np.concatenate((sigmas_t1, sigmas_t2[len(sigmas_t1):]), axis=0)
    rhos_t = np.concatenate((rhos_t1, rhos_t2[len(rhos_t1):]), axis=0)
    fitting_data = np.concatenate((train_data, border_centers), axis=0)
    if edge_to is None:
        edge_to = edge_to_t
        edge_from = edge_from_t
        weight = weight_t
        probs = weight_t / weight_t.max()
        feature_vectors = fitting_data
        sigmas = sigmas_t
        rhos = rhos_t
        knn_indices = knn_idxs_t
        n_vertices = len(train_data)
    else:
        # every round, we need to add len(data) to edge_to(as well as edge_from) index
        increase_idx = t * len(fitting_data)
        edge_to = np.concatenate((edge_to, edge_to_t + increase_idx), axis=0)
        edge_from = np.concatenate((edge_from, edge_from_t + increase_idx), axis=0)
        # normalize weight to be in range (0, 1)
        weight = np.concatenate((weight, weight_t), axis=0)
        probs_t = weight_t / weight_t.max()
        probs = np.concatenate((probs, probs_t), axis=0)
        sigmas = np.concatenate((sigmas, sigmas_t), axis=0)
        rhos = np.concatenate((rhos, rhos_t), axis=0)
        feature_vectors = np.concatenate((feature_vectors, fitting_data), axis=0)
        knn_indices = np.concatenate((knn_indices, knn_idxs_t+increase_idx), axis=0)

heads, tails, vals = construct_temporal_edge_dataset(X=feature_vectors,
                                                        n_vertices=n_vertices,
                                                        persistent=TEMPORAL_PERSISTANT,
                                                        time_steps=TIME_STEPS,
                                                        knn_indices=knn_indices,
                                                        sigmas=sigmas,
                                                        rhos=rhos
                                                        )
weight = np.concatenate((weight, vals), axis=0)
probs_t = vals / vals.max()
probs = np.concatenate((probs, probs_t), axis=0)
edge_to = np.concatenate((edge_to, heads), axis=0)
edge_from = np.concatenate((edge_from, tails), axis=0)

dataset = DataHandler(edge_to, edge_from, feature_vectors)


result = np.zeros(weight.shape[0], dtype=np.float64)
n_samples = np.sum(NUMS * probs)

sampler = WeightedRandomSampler(probs, n_samples, replacement=False)
edge_loader = DataLoader(dataset, batch_size=1000, sampler=sampler)


trainer = SingleVisTrainer(model, criterion, optimizer, edge_loader=edge_loader, DEVICE=DEVICE)
patient = PATIENT
for epoch in EPOCH_NUMS:
    prev_loss = trainer.loss
    loss = trainer.train_step()
    # early stop, check whether converge or not
    if prev_loss - loss < 1E-2:
        if patient == 0:
            break
        else:
            patient -= 1
    else:
        patient = PATIENT

trainer.save(name="cifar10_epoch_10")

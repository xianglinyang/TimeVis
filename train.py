import torch
import sys
import os
import numpy as np
import time
import argparse

from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from umap.umap_ import find_ab_params

from singleVis.custom_weighted_random_sampler import CustomWeightedRandomSampler
from singleVis.SingleVisualizationModel import SingleVisualizationModel
from singleVis.losses import SingleVisLoss, UmapLoss, ReconstructionLoss
from singleVis.edge_dataset import DataHandler
from singleVis.trainer import SingleVisTrainer
from singleVis.data import DataProvider
from singleVis.backend import fuzzy_complex, boundary_wise_complex, construct_step_edge_dataset, \
    construct_temporal_edge_dataset, get_attention
import singleVis.config as config


parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--content_path', type=str)
parser.add_argument('-d','--dataset', choices=['online','cifar10', 'mnist', 'fmnist'])

args = parser.parse_args()

CONTENT_PATH = args.content_path
DATASET = args.dataset
LEN = config.dataset_config[DATASET]["TRAINING_LEN"]
LAMBDA = config.dataset_config[DATASET]["LAMBDA"]

# define hyperparameters

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
EPOCH_NUMS = config.dataset_config[DATASET]["training_config"]["EPOCH_NUM"]
TIME_STEPS = config.dataset_config[DATASET]["training_config"]["TIME_STEPS"]
TEMPORAL_PERSISTENT = config.dataset_config[DATASET]["training_config"]["TEMPORAL_PERSISTENT"]
NUMS = config.dataset_config[DATASET]["training_config"]["NUMS"]    # how many epoch should we go through for one pass
PATIENT = config.dataset_config[DATASET]["training_config"]["PATIENT"]

content_path = CONTENT_PATH
sys.path.append(content_path)

from Model.model import *
net = resnet18()
classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
selected_idxs = np.random.choice(np.arange(LEN), size=3000, replace=False)

data_provider = DataProvider(content_path, net, 1, TIME_STEPS, 1, split=-1, device=DEVICE, verbose=1)
# data_provider.initialize(LEN//10, l_bound=0.6)

time_start = time.time()
model = SingleVisualizationModel(input_dims=512, output_dims=2, units=256)
negative_sample_rate = 5
min_dist = .1
_a, _b = find_ab_params(1.0, min_dist)
umap_loss_fn = UmapLoss(negative_sample_rate, DEVICE, _a, _b, repulsion_strength=1.0)
recon_loss_fn = ReconstructionLoss(beta=1.0)
criterion = SingleVisLoss(umap_loss_fn, recon_loss_fn, lambd=LAMBDA)

optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)


# TODO construct spatio-temporal complex and get edges
# dummy input
edge_to = None
edge_from = None
sigmas = None
rhos = None
weight = None
probs = None
feature_vectors = None
attention = None
knn_indices = None
time_steps_num = list()

# each time step

for t in range(1, TIME_STEPS+1, 1):
    # load train data and border centers
    train_data = data_provider.train_representation(t).squeeze()
    # selected_idxs = selected_idxs[:int(0.9*len(selected_idxs))]
    # train_data = train_data[selected_idxs]
    border_centers = data_provider.border_representation(t).squeeze()
    border_centers = border_centers

    complex, sigmas_t1, rhos_t1, knn_idxs_t = fuzzy_complex(train_data, 15)
    bw_complex, sigmas_t2, rhos_t2, _ = boundary_wise_complex(train_data, border_centers, 15)
    edge_to_t, edge_from_t, weight_t = construct_step_edge_dataset(complex, bw_complex, NUMS)
    sigmas_t = np.concatenate((sigmas_t1, sigmas_t2[len(sigmas_t1):]), axis=0)
    rhos_t = np.concatenate((rhos_t1, rhos_t2[len(rhos_t1):]), axis=0)
    fitting_data = np.concatenate((train_data, border_centers), axis=0)
    pred_model = data_provider.prediction_function(t)
    attention_t = get_attention(pred_model, fitting_data, temperature=.01, device=DEVICE, verbose=1)
    # increase_idx = (t-1) * int(len(train_data) * 1.1)
    t_num = len(train_data)
    b_num = len(border_centers)
    if edge_to is None:
        edge_to = edge_to_t
        edge_from = edge_from_t
        weight = weight_t
        probs = weight_t / weight_t.max()
        feature_vectors = fitting_data
        attention = attention_t
        sigmas = sigmas_t
        rhos = rhos_t
        knn_indices = knn_idxs_t
        time_steps_num.append((t_num, b_num))
    else:
        # every round, we need to add len(data) to edge_to(as well as edge_from) index
        increase_idx = len(feature_vectors)
        edge_to = np.concatenate((edge_to, edge_to_t + increase_idx), axis=0)
        edge_from = np.concatenate((edge_from, edge_from_t + increase_idx), axis=0)
        # normalize weight to be in range (0, 1)
        weight = np.concatenate((weight, weight_t), axis=0)
        probs_t = weight_t / weight_t.max()
        probs = np.concatenate((probs, probs_t), axis=0)
        sigmas = np.concatenate((sigmas, sigmas_t), axis=0)
        rhos = np.concatenate((rhos, rhos_t), axis=0)
        feature_vectors = np.concatenate((feature_vectors, fitting_data), axis=0)
        attention = np.concatenate((attention, attention_t), axis=0)
        knn_indices = np.concatenate((knn_indices, knn_idxs_t+increase_idx), axis=0)
        time_steps_num.append((t_num, b_num))

# boundary points...
heads, tails, vals = construct_temporal_edge_dataset(X=feature_vectors,
                                                     time_step_nums=time_steps_num,
                                                     persistent=TEMPORAL_PERSISTENT,
                                                     time_steps=TIME_STEPS,
                                                     knn_indices=knn_indices,
                                                     sigmas=sigmas,
                                                     rhos=rhos)
# remove elements with very low probability
eliminate_idxs = (vals < 1e-2)
heads = heads[eliminate_idxs]
tails = tails[eliminate_idxs]
vals = vals[eliminate_idxs]
# increase weight of temporal edges
vals = vals*80

weight = np.concatenate((weight, vals), axis=0)
probs_t = vals / (vals.max() + 1e-4)
probs = np.concatenate((probs, probs_t), axis=0)
edge_to = np.concatenate((edge_to, heads), axis=0)
edge_from = np.concatenate((edge_from, tails), axis=0)

dataset = DataHandler(edge_to, edge_from, feature_vectors, attention)


# result = np.zeros(weight.shape[0], dtype=np.float64)
n_samples = int(np.sum(NUMS * probs) // 1)

sampler = WeightedRandomSampler(probs, n_samples, replacement=True)
edge_loader = DataLoader(dataset, batch_size=1000, sampler=sampler)

trainer = SingleVisTrainer(model, criterion, optimizer, edge_loader=edge_loader, DEVICE=DEVICE)
patient = PATIENT
for epoch in range(EPOCH_NUMS):
    print("====================\nepoch:{}\n===================".format(epoch))
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

time_end = time.time()
time_spend = time_end - time_start
print("Time spend: {:.2f}".format(time_spend))
trainer.save(save_dir=data_provider.model_path, file_name="SV")
trainer.load(file_path=os.path.join(data_provider.model_path,"SV.pth"))


########################################################################################################################
# evaluate
########################################################################################################################
from singleVis.eval.evaluator import Evaluator
evaluator = Evaluator(data_provider, trainer)
evaluator.save_eval(n_neighbors=15, file_name="evaluation")

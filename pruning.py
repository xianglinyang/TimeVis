import torch
import sys
import os
import numpy as np
from umap.umap_ import find_ab_params
import argparse

from singleVis.data import DataProvider
from singleVis.backend import prune_points
from singleVis.utils import knn, hausdorff_dist

import singleVis.config as config
parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--content_path', type=str)
parser.add_argument('-d','--dataset', choices=['cifar10', 'mnist', 'fmnist'])

args = parser.parse_args()
CONTENT_PATH = args.content_path
DATASET = args.dataset

LEN = config.dataset_config[DATASET]["TRAINING_LEN"]
LAMBDA = config.dataset_config[DATASET]["LAMBDA"]
DOWNSAMPLING_RATE = config.dataset_config[DATASET]["DOWNSAMPLING_RATE"]
L_BOUND = config.dataset_config[DATASET]["L_BOUND"]

# define hyperparameters

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
EPOCH_NUMS = config.dataset_config[DATASET]["training_config"]["EPOCH_NUM"]
TIME_STEPS = config.dataset_config[DATASET]["training_config"]["TIME_STEPS"]
TEMPORAL_PERSISTENT = config.dataset_config[DATASET]["training_config"]["TEMPORAL_PERSISTENT"]
NUMS = config.dataset_config[DATASET]["training_config"]["NUMS"]    # how many epoch should we go through for one pass
PATIENT = config.dataset_config[DATASET]["training_config"]["PATIENT"]
TEMPORAL_EDGE_WEIGHT = config.dataset_config[DATASET]["training_config"]["TEMPORAL_EDGE_WEIGHT"]

content_path = CONTENT_PATH
sys.path.append(content_path)
from Model.model import *
net = resnet18()
classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

data_provider = DataProvider(content_path, net, 1, TIME_STEPS, 1, split=-1, device=DEVICE, verbose=1)

train_data = data_provider.train_representation(1)
selected_idxs = np.random.choice(np.arange(len(train_data)), size=len(train_data) // 5, replace=False)
while len(train_data) > 5000:
    knn_idxs, _ = knn(train_data, k=15)
    selected_idxs = prune_points(knn_idxs, 10, threshold=0.5)
    if len(selected_idxs) < 200:
        break
    remain_idxs = [i for i in range(len(knn_idxs)) if i not in selected_idxs]
    train_data = train_data[remain_idxs]
train_data = data_provider.train_representation(1)
hausdorff, _ = hausdorff_dist(train_data, remain_idxs, n_neighbors=15)
print("hausdorff distance: {:.2f}".format(hausdorff))

# from sklearn import utils
import torch
import sys
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from umap.umap_ import find_ab_params
import time
import json

from singleVis.SingleVisualizationModel import SingleVisualizationModel
from singleVis.losses import SingleVisLoss, UmapLoss, ReconstructionLoss
from singleVis.edge_dataset import DataHandler
from singleVis.trainer import SingleVisTrainer
from singleVis.data import DataProvider

from singleVis.backend import fuzzy_complex, boundary_wise_complex, construct_step_edge_dataset, \
    construct_temporal_edge_dataset, get_attention, construct_temporal_edge_dataset2
import singleVis.config as config
import argparse
from singleVis import utils
parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--content_path', type=str)
parser.add_argument('-d','--dataset', choices=['cifar10', 'mnist', 'fmnist'])

args = parser.parse_args()

CONTENT_PATH = args.content_path
DATASET = args.dataset
LEN = config.dataset_config[DATASET]["TRAINING_LEN"]
LAMBDA = config.dataset_config[DATASET]["LAMBDA"]

# define hyperparameters

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCH_NUMS = config.training_config["EPOCH_NUM"]
TIME_STEPS = config.training_config["TIME_STEPS"]
TEMPORAL_PERSISTENT = config.training_config["TEMPORAL_PERSISTENT"]
NUMS = config.training_config["NUMS"]    # how many epoch should we go through for one pass
PATIENT = config.training_config["PATIENT"]

content_path = CONTENT_PATH
sys.path.append(content_path)

from Model.model import *
net = resnet18()
classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

data_provider = DataProvider(content_path, net, 1, TIME_STEPS, 1, split=-1, device=DEVICE, verbose=1)


# each time step

ratio_1 = np.arange(1,10,1)/10.
ratio_2 = np.arange(1,10,1)/100.
ratio = np.concatenate((ratio_2, ratio_1), axis=0)
eval = dict()
for stage in [1,4,7]:
    eval[stage] = dict()
    for r in ratio:
        eval[stage][r] = dict()
        mean_list = list()
        t = list()
        for _ in range(10):
            selected_idxs = np.random.choice(np.arange(LEN), size=int(LEN*r), replace=False)
            train_data = data_provider.train_representation(stage).squeeze()
            hausdorff_, t_ = utils.hausdorff_dist(train_data, selected_idxs, n_neighbors=15)
            mean_list.append(hausdorff_)
            t.append(t_)
        print("{:.2f}:{:.3f}".format(r, np.array(mean_list).mean()))
        eval[stage][r]["mean"] = np.array(mean_list).mean()
        eval[stage][r]["std"] = np.array(mean_list).std()
        eval[stage][r]["time"] = np.array(t).mean()
        eval[stage][r]["time_std"] = np.array(t).std()
with open("hausdorff_{}.json".format(DATASET), "w") as f:
    json.dump(eval, f)



        


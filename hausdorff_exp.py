# from sklearn import utils
import torch
import sys
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from umap.umap_ import find_ab_params
import json

from singleVis.SingleVisualizationModel import SingleVisualizationModel
from singleVis.losses import SingleVisLoss, UmapLoss, ReconstructionLoss
from singleVis.edge_dataset import DataHandler
from singleVis.trainer import SingleVisTrainer
from singleVis.data import DataProvider

import singleVis.config as config
import argparse
from singleVis import utils
from singleVis import kcenter_greedy

parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--content_path', type=str)
parser.add_argument('-d','--dataset', choices=['online','cifar10', 'mnist', 'fmnist'])
parser.add_argument('-p',"--preprocess", choices=[0,1], default=0)
parser.add_argument('-g',"--gpu_id", type=int, choices=[0,1,2,3], default=0)
args = parser.parse_args()

CONTENT_PATH = args.content_path
DATASET = args.dataset
PREPROCESS = args.preprocess
GPU_ID = args.gpu_id

LEN = config.dataset_config[DATASET]["TRAINING_LEN"]
LAMBDA = config.dataset_config[DATASET]["LAMBDA"]
DOWNSAMPLING_RATE = config.dataset_config[DATASET]["DOWNSAMPLING_RATE"]
L_BOUND = config.dataset_config[DATASET]["L_BOUND"]

# define hyperparameters

DEVICE = torch.device("cuda:{:d}".format(GPU_ID) if torch.cuda.is_available() else "cpu")
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

# each time step
ratio = np.array([0.9, 0.1,0.01])
# ratio_1 = np.arange(1,10,1)/10.
# ratio_2 = np.arange(1,10,1)/100.
# ratio = np.concatenate((ratio_2, ratio_1), axis=0)

eval = dict()
for stage in [1,5,10]:
    eval[stage] = dict()
    for r in ratio:
        eval[stage][r] = dict()
        mean_list = list()
        t = list()
        train_data = data_provider.train_representation(stage).squeeze()
        for _ in range(1):
            selected_idxs = np.random.choice(np.arange(LEN), size=int(LEN*r), replace=False)
            hausdorff_, t_ = utils.hausdorff_dist_cus(train_data, selected_idxs)
            mean_list.append(hausdorff_)
            t.append(t_)
        print("{:.2f}:{:.3f}".format(r, np.array(mean_list).mean()))
        eval[stage][r]["mean"] = np.array(mean_list).mean()
        eval[stage][r]["std"] = np.array(mean_list).std()
        eval[stage][r]["time"] = np.array(t).mean()
        eval[stage][r]["time_std"] = np.array(t).std()
    
    # kc
    selected_idxs = np.random.choice(np.arange(LEN), size=100, replace=False)
    target_num = int(LEN*0.01)
    kc = kcenter_greedy.kCenterGreedy(train_data)
    _, dists = kc.select_batch_with_budgets(selected_idxs, target_num-100, return_min=True)
    eval[stage][0.01]["kc"] = dists
with open("hausdorff_{}.json".format(DATASET), "w") as f:
    json.dump(eval, f)



        


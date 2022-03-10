# from sklearn import utils
import torch
import sys
import numpy as np
import json

from singleVis.data import DataProvider

import singleVis.config as config
import argparse
from singleVis import utils
from singleVis import kcenter_greedy

parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--content_path', type=str)
parser.add_argument('-d','--dataset', choices=['online','cifar10', 'mnist', 'fmnist', 'cifar10_full', 'mnist_full', 'fmnist_full'])
parser.add_argument('-p',"--preprocess", choices=[0,1], default=0)
parser.add_argument('-g',"--gpu_id", type=int, choices=[0,1,2,3], default=0)
args = parser.parse_args()

CONTENT_PATH = args.content_path
DATASET = args.dataset
PREPROCESS = args.preprocess
GPU_ID = args.gpu_id

LEN = config.dataset_config[DATASET]["TRAINING_LEN"]
LAMBDA = config.dataset_config[DATASET]["LAMBDA"]
L_BOUND = config.dataset_config[DATASET]["L_BOUND"]
MAX_HAUSDORFF = config.dataset_config[DATASET]["MAX_HAUSDORFF"]
ALPHA = config.dataset_config[DATASET]["ALPHA"]
BETA = config.dataset_config[DATASET]["BETA"]
INIT_NUM = config.dataset_config[DATASET]["INIT_NUM"]
EPOCH_START = config.dataset_config[DATASET]["EPOCH_START"]
EPOCH_END = config.dataset_config[DATASET]["EPOCH_END"]
EPOCH_PERIOD = config.dataset_config[DATASET]["EPOCH_PERIOD"]

# define hyperparameters
DEVICE = torch.device("cuda:{:d}".format(GPU_ID) if torch.cuda.is_available() else "cpu")
S_N_EPOCHS = config.dataset_config[DATASET]["training_config"]["S_N_EPOCHS"]
B_N_EPOCHS = config.dataset_config[DATASET]["training_config"]["B_N_EPOCHS"]
T_N_EPOCHS = config.dataset_config[DATASET]["training_config"]["T_N_EPOCHS"]
N_NEIGHBORS = config.dataset_config[DATASET]["training_config"]["N_NEIGHBORS"]
PATIENT = config.dataset_config[DATASET]["training_config"]["PATIENT"]
MAX_EPOCH = config.dataset_config[DATASET]["training_config"]["MAX_EPOCH"]

content_path = CONTENT_PATH
sys.path.append(content_path)

from Model.model import *
net = resnet18()
classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


data_provider = DataProvider(content_path, net, EPOCH_START, EPOCH_END, EPOCH_PERIOD, split=-1, device=DEVICE, verbose=1)

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



        


from os import replace
import numpy as np
import scipy
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pynndescent import NNDescent

import time
import math
from sklearn.metrics.pairwise import KERNEL_PARAMS
import torch
import sys
import argparse
from tqdm import tqdm

from sklearn.metrics import pairwise_distances
from singleVis.data import DataProvider
from singleVis.backend import construct_spatial_temporal_complex, select_points_step
import singleVis.config as config
from singleVis.kcenter_greedy import kCenterGreedy
from singleVis.utils import hausdorff_dist_cus


def find_mu(data):
    # number of trees in random projection forest
    n_trees = min(64, 5 + int(round((data.shape[0]) ** 0.5 / 20.0)))
    # max number of nearest neighbor iters to perform
    n_iters = max(5, int(round(np.log2(data.shape[0]))))
    # distance metric
    metric = "euclidean"
    # get nearest neighbors
    nnd = NNDescent(
        data,
        n_neighbors=3,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=False
    )
    _, knn_dists = nnd.neighbor_graph
    mu = knn_dists[:, 2] / (knn_dists[:, 1]+1e-4) + 1e-4
    return mu

def twonn_dimension_fast(data):
    N = len(data)
    mu = find_mu(data).tolist()
    mu = list(enumerate(mu, start=1))  
    sigma_i = dict(zip(range(1,len(mu)+1), np.array(sorted(mu, key=lambda x: x[1]))[:,0].astype(int)))
    mu = dict(mu)
    F_i = {}
    for i in mu:
        F_i[sigma_i[i]] = i/N
    x = np.log([mu[i] for i in sorted(mu.keys())])
    y = np.array([1-F_i[i] for i in sorted(mu.keys())])
    x = x[y>0]
    y = y[y>0]
    y = -1*np.log(y)
    d = np.linalg.lstsq(np.vstack([x, np.zeros(len(x))]).T, y, rcond=None)[0][0]
    return d

if __name__ == "__main__":

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

    idxs_ = np.random.choice(np.arange(LEN), size=300, replace=False)

    data = data_provider.train_representation(TIME_STEPS)
    max_x = np.linalg.norm(data, axis=1).max()
    data = data/max_x

    kc = kCenterGreedy(data)
    d0 = twonn_dimension_fast(data)
    _ = kc.select_batch_with_budgets(idxs_,100)
    all_idxs = kc.already_selected
    haus = kc.hausdorff()
    c0 = haus

    
    for i in [TIME_STEPS, 5,4,3,2,1]:
        t0 = time.time()
        print("============={:d}================".format(i))
        data = data_provider.train_representation(i)
        max_x = np.linalg.norm(data, axis=1).max()
        data = data/max_x
        d = twonn_dimension_fast(data)
        kc = kCenterGreedy(data)
        # _ = kc.select_batch_with_cn(idxs,0.1, p=0.95)

        _ = kc.select_batch_with_budgets(idxs_, 100)
        haus = kc.hausdorff()
        print(haus,haus/c0, d/d0)
        idxs = kc.already_selected
        ratio = haus/c0
        _ = kc.select_batch_with_budgets(idxs, 600)
        haus = kc.hausdorff()
        print(haus,haus/ratio)
        idxs = kc.already_selected
        _ = kc.select_batch_with_budgets(idxs, 1000)
        haus = kc.hausdorff()
        print(haus,haus/ratio)
        idxs = kc.already_selected
        _ = kc.select_batch_with_budgets(idxs, 1000)
        haus = kc.hausdorff()
        print(haus,haus/ratio)
        # _ = kc.select_batch_with_cn(idxs, 0.1 , d_0, p=.80)
        # # _ = kc.select_batch_with_cn(idxs, 5.5*c*math.pow(d/d_0,(d/d_0)), p=0.95)
        # # new_batch = kc.select_batch_with_cn(idxs, 0.04/d)
        
        t1= time.time()
        print(t1-t0)


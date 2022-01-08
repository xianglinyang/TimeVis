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

def get_unit(data, init_num=200, adding_num=100):
    t0 = time.time()
    l = len(data)
    idxs = np.random.choice(np.arange(l), size=init_num, replace=False)
    # _,_ = hausdorff_dist_cus(data, idxs)
    kc = kCenterGreedy(data)
    d0 = twonn_dimension_fast(data)
    _ = kc.select_batch_with_budgets(idxs, adding_num)
    c0 = kc.hausdorff()
    t1 = time.time()
    return c0, d0, "{:.1f}".format(t1-t0)
    

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

    # init with 100,200,300
    # adding 100,200 points to get a initial result

    # init with 200, adding 100 points
    # data = data_provider.train_representation(TIME_STEPS)
    # max_x = np.linalg.norm(data, axis=1).max()
    # data = data/max_x
    # c0, d0, t0 = get_unit(data)
    # print("find ratio in {} seconds".format(t0))
    
    # idxs_ = np.random.choice(np.arange(LEN), size=300, replace=False)
    # for i in range(TIME_STEPS, 0, -1):
    #     print("=========================={:d}==========================".format(i))
    #     data = data_provider.train_representation(i)
    #     max_x = np.linalg.norm(data, axis=1).max()
    #     data = data/max_x
    #     c, d, t = get_unit(data)
    #     print(c/c0, d/d0)
    #     ratio = c/c0

    #     kc = kCenterGreedy(data)
    #     _ = kc.select_batch_with_budgets(idxs_, 5700)
    #     haus = kc.hausdorff()
    #     idxs = kc.already_selected
    #     print(haus, haus/ratio, haus/ratio/d*d0)

        # _ = kc.select_batch_with_budgets(idxs, 1200)
        # haus = kc.hausdorff()
        # idxs = kc.already_selected
        # print(haus, haus/ratio)

        # _ = kc.select_batch_with_budgets(idxs, 1000)
        # haus = kc.hausdorff()
        # idxs = kc.already_selected
        # print(haus, haus/ratio/d*d0)
    if DATASET == "fmnist":
        # 232s
        alpha = .5
        beta = 1
        threshold = 0.07
    elif DATASET == "cifar10":
        # 124s
        alpha = 0
        beta = 1
        threshold = 0.2
    else:
        # mnist
        # 208.6s
        alpha = .5
        beta = 1
        threshold = 0.195

    train_num = data_provider.train_num
    selected_idxs = np.random.choice(np.arange(train_num), size=int(train_num * 0.005), replace=False)

    baseline_data = data_provider.train_representation(TIME_STEPS)
    max_x = np.linalg.norm(baseline_data, axis=1).max()
    baseline_data = baseline_data/max_x
    
    c0,d0,_ = get_unit(baseline_data)

    # each time step
    t0 = time.time()
    for t in range(TIME_STEPS, 0, -1):
        print("================{:d}=================".format(t))
        # load train data and border centers
        train_data = data_provider.train_representation(t).squeeze()

        # normalize data by max ||x||_2
        max_x = np.linalg.norm(train_data, axis=1).max()
        train_data = train_data/max_x

        # get normalization parameters for different epochs
        c,d,_ = get_unit(train_data)
        c_c0 = math.pow(c/c0, beta)
        d_d0 = math.pow(d/d0, alpha)
        print("Finish calculating normaling factor")

        kc = kCenterGreedy(train_data)
        _ = kc.select_batch_with_cn(selected_idxs, threshold, c_c0, d_d0, p=0.95)
        selected_idxs = kc.already_selected.astype("int")
        print("select {:d} points".format(len(selected_idxs)))
    t1 = time.time()
    print("Selecting points takes {:.1f} seconds".format(t1-t0))





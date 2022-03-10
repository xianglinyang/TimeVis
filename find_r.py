import numpy as np
from pynndescent import NNDescent

import time
import math
import torch
import sys
import argparse

from singleVis.data import DataProvider
import singleVis.config as config
from singleVis.kcenter_greedy import kCenterGreedy


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

    if DATASET == "fmnist":
        # 232s
        # alpha = 2
        # beta = 1.3
        # threshold = 0.06
        alpha = 2
        beta = 1.3
        threshold = 0.06
    elif DATASET == "cifar10":
        # 124s
        # alpha = 0
        # beta = .1
        # threshold = 0.2
        alpha = 0.
        beta = .1
        threshold = 0.18
    else:
        # mnist
        # 208.6s
        alpha = 1.#1.5
        beta = 1
        threshold = 0.25

    train_num = data_provider.train_num
    selected_idxs = np.random.choice(np.arange(train_num), size=300, replace=False)

    baseline_data = data_provider.train_representation(EPOCH_END)
    max_x = np.linalg.norm(baseline_data, axis=1).max()
    baseline_data = baseline_data/max_x
    
    c0,d0,_ = get_unit(baseline_data)

    # each time step
    t0 = time.time()
    for t in range(EPOCH_END, EPOCH_START-1, -EPOCH_PERIOD):
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


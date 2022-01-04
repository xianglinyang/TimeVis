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
from singleVis.backend import construct_spatial_temporal_complex
import singleVis.config as config


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
    mu = knn_dists[:, 2] / knn_dists[:, 1]
    return mu

def estimate_id_fast(data):
    mu = find_mu(data)
    N = data.shape[0]
    sort_idx = np.argsort(mu)
    Femp     = np.arange(N)/N
    lr = LinearRegression(fit_intercept=False)
    lr.fit(np.log(mu[sort_idx]).reshape(-1,1), -np.log(1-Femp).reshape(-1,1))
    d = lr.coef_[0][0] 
    return d
    
def estimate_id(data):
    N = data.shape[0]
    mu = np.zeros(N)
    for i in tqdm(range(N)):
        dist = np.sort(np.sqrt(np.sum((data[i]-data)**2, axis=1)))
        r1, r2 = dist[dist>0][:2]
        mu[i]=r2/r1
    sort_idx = np.argsort(mu)
    Femp     = np.arange(N)/N
    lr = LinearRegression(fit_intercept=False)
    lr.fit(np.log(mu[sort_idx]).reshape(-1,1), -np.log(1-Femp).reshape(-1,1))
    d = lr.coef_[0][0] 

    return d

def twonn_dimension(data, return_xy=False):
    N = len(data)
    mu = []
    for i in tqdm(range(N)):
        dist = np.sort(np.sqrt(np.sum((data[i]-data)**2, axis=1)))
        r1, r2 = dist[dist>0][:2]
        mu.append((i+1,r2/r1))
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
    if return_xy:
        return d, x, y
    else: 
        return d

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
    for i in range(1, TIME_STEPS+1, 1):
        data = data_provider.train_representation(i)
        # t0 = time.time()
        # d1 = estimate_id(data)
        # t1 = time.time()
        # d2 = estimate_id_fast(data)
        # t2 = time.time()
        # d3 = twonn_dimension(data)
        # t3 = time.time()
        # d4 = twonn_dimension_fast(data)
        # t4 = time.time()
        # print("{}-th iteration, {:.2f} dim with in {:.2f} seconds, {:.2f} dim with in {:.2f} seconds,".format(i, d1, t1-t0,d2, t2-t1))
        # print("{}-th iteration, {:.2f} dim with in {:.2f} seconds, {:.2f} dim with in {:.2f} seconds,".format(i, d3, t3-t2,d4, t4-t3))
        c_all = np.linalg.norm(data, axis=1)
        c = c_all.max()
        print(c)
        t0 = time.time()
        d = twonn_dimension_fast(data)
        t1 = time.time()
        num = math.pow(2*c*math.sqrt(d)/23.2, int(d))
        print("{}-th  with {:.2f} dim in {:.2f} seconds need num {:.0f}".format(i, d, t1-t0, num))
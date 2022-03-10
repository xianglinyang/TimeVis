import numpy as np
from sklearn.linear_model import LinearRegression
from pynndescent import NNDescent

import time
import math
import torch
import sys
import argparse
from tqdm import tqdm

from singleVis.data import DataProvider
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

    for i in range(EPOCH_START, EPOCH_END+1, EPOCH_PERIOD):
        data = data_provider.train_representation(i)
        c_all = np.linalg.norm(data, axis=1)
        c = c_all.mean()
        print(c_all.max(), c_all.min(), c_all.mean())
        t0 = time.time()
        d = twonn_dimension_fast(data)
        t1 = time.time()
        num = math.pow(2*c*math.sqrt(d)/100, int(d))
        print("{}-th  with {:.2f} dim in {:.2f} seconds {:.0f}".format(i, d, t1-t0, num))
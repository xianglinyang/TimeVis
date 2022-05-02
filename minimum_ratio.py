from tkinter import NE
from matplotlib.pyplot import axis
import torch
import sys
import os
import json
import time
import numpy as np
import argparse
import multiprocessing as mp
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from umap.umap_ import find_ab_params

from singleVis.custom_weighted_random_sampler import CustomWeightedRandomSampler
from singleVis.SingleVisualizationModel import SingleVisualizationModel
from singleVis.losses import SingleVisLoss, UmapLoss, ReconstructionLoss
from singleVis.edge_dataset import DataHandler
from singleVis.trainer import SingleVisTrainer
from singleVis.data import DataProvider
import singleVis.config as config
from singleVis.eval.evaluator import Evaluator
from singleVis.spatial_edge_constructor import kcParallelSpatialEdgeConstructor
from singleVis.temporal_edge_constructor import GlobalParallelTemporalEdgeConstructor

# CONTENT_PATH = "/home/xianglin/projects/DVI_data/resnet18_cifar10"
# DATASET = "cifar10_full"
CONTENT_PATH = "/home/xianglin/projects/DVI_data/resnet18_mnist"
DATASET = "mnist_full"
CONTENT_PATH = "/home/xianglin/projects/DVI_data/resnet18_fmnist"
DATASET = "fmnist_full"
PREPROCESS = 0
GPU_ID = 2

NET = config.dataset_config[DATASET]["NET"]
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

# from Model.model import *
import Model.model as subject_model
# net = resnet18()
net = eval("subject_model.{}()".format(NET))
classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

save_dir = os.path.join(content_path, "minimum_r")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# epoch_list = [20,30,40,60,80,100, 120, 140,160,180,200]
# max_haus_list = [0.4, 0.3, 0.28,0.26,0.24,0.21,0.18,0.15,0.1]
# epoch_list=[180,200]
# max_haus_list = [0.3,0.25]
# epoch_list = [1,5,10,15,20]
# max_haus_list = [0.6, 0.5, 0.4, 0.35, 0.3, 0.28,0.25,0.21,0.18,0.15]
epoch_list = [1,5,10,20,30,40,50]
max_haus_list = [.15, .1, .07, .05, .03, .01]
for epoch in epoch_list:
    nums = list()
    nn_train_15 = list()
    nn_test_15 = list()
    b_train_15 = list()
    b_test_15 = list()
    ppr_train = list()
    ppr_test = list()

    data_provider = DataProvider(content_path, net, epoch, epoch, EPOCH_PERIOD, split=-1, device=DEVICE, verbose=1)
    data = data_provider.train_representation(epoch)
    max_x = np.linalg.norm(data, axis=1).max()
    for max_haus in max_haus_list:

        ########################################################################################################################
        #                                                    TRAINING SETTING                                                  #
        ########################################################################################################################
        
        model = SingleVisualizationModel(input_dims=512, output_dims=2, units=256)
        negative_sample_rate = 5
        min_dist = .1
        _a, _b = find_ab_params(1.0, min_dist)
        umap_loss_fn = UmapLoss(negative_sample_rate, DEVICE, _a, _b, repulsion_strength=1.0)
        recon_loss_fn = ReconstructionLoss(beta=1.0)
        criterion = SingleVisLoss(umap_loss_fn, recon_loss_fn, lambd=LAMBDA)

        optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)


        t0 = time.time()
        spatial_cons = kcParallelSpatialEdgeConstructor(data_provider=data_provider, init_num=INIT_NUM, s_n_epochs=S_N_EPOCHS, b_n_epochs=B_N_EPOCHS, n_neighbors=N_NEIGHBORS, MAX_HAUSDORFF=max_haus, ALPHA=ALPHA, BETA=BETA)
        s_edge_to, s_edge_from, s_probs, feature_vectors, time_step_nums, selected_idxs_list, knn_indices, sigmas, rhos, attention = spatial_cons.construct()

        temporal_cons = GlobalParallelTemporalEdgeConstructor(X=feature_vectors, time_step_nums=time_step_nums, sigmas=sigmas, rhos=rhos, n_neighbors=N_NEIGHBORS, n_epochs=T_N_EPOCHS, selected_idxs_lists=selected_idxs_list)
        t_edge_to, t_edge_from, t_probs = temporal_cons.construct()
        t1 = time.time()

        edge_to = np.concatenate((s_edge_to, t_edge_to),axis=0)
        edge_from = np.concatenate((s_edge_from, t_edge_from), axis=0)
        probs = np.concatenate((s_probs, t_probs), axis=0)
        probs = probs / (probs.max()+1e-3)
        eliminate_zeros = probs>1e-3
        edge_to = edge_to[eliminate_zeros]
        edge_from = edge_from[eliminate_zeros]
        probs = probs[eliminate_zeros]

        dataset = DataHandler(edge_to, edge_from, feature_vectors, attention)
        n_samples = int(np.sum(S_N_EPOCHS * probs) // 1)
        # chosse sampler based on the number of dataset
        if len(edge_to) > 2^24:
            sampler = CustomWeightedRandomSampler(probs, n_samples, replacement=True)
        else:
            sampler = WeightedRandomSampler(probs, n_samples, replacement=True)
        edge_loader = DataLoader(dataset, batch_size=1000, sampler=sampler)
        trainer = SingleVisTrainer(model, criterion, optimizer, lr_scheduler,edge_loader=edge_loader, DEVICE=DEVICE)
        trainer.train(PATIENT, MAX_EPOCH)
        evaluator = Evaluator(data_provider, trainer)

        nums.append(selected_idxs_list[0].shape[0])
        nn_train_15.append(evaluator.eval_nn_train(epoch,15))
        nn_test_15.append(evaluator.eval_nn_test(epoch, 15))
        b_train_15.append(evaluator.eval_b_train(epoch,15))
        b_test_15.append(evaluator.eval_b_test(epoch, 15))
        ppr_train.append(evaluator.eval_inv_train(epoch))
        ppr_test.append(evaluator.eval_inv_test(epoch))
    
    # draw plot
    l1 = plt.plot(nums, nn_train_15, "ro-", label="nn_train")
    l2 = plt.plot(nums, nn_test_15, "r+-", label="nn_test")
    l3 = plt.plot(nums, ppr_train, "bo-", label="ppr_train")
    l4 = plt.plot(nums, ppr_test, "b+-", label="ppr_test")
    l5 = plt.plot(nums, b_train_15, "go-", label="b_train")
    l6 = plt.plot(nums, b_test_15, "g+-", label="b_test")
    plt.title("nums")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "{}_nums".format(epoch)))
    plt.clf()
    l1 = plt.plot(max_haus_list, nn_train_15, "ro-", label="nn_train")
    l2 = plt.plot(max_haus_list, nn_test_15, "r+-", label="nn_test")
    l3 = plt.plot(max_haus_list, ppr_train, "bo-", label="ppr_train")
    l4 = plt.plot(max_haus_list, ppr_test, "b+-", label="ppr_test")
    l5 = plt.plot(max_haus_list, b_train_15, "go-", label="b_train")
    l6 = plt.plot(max_haus_list, b_test_15, "g+-", label="b_test")
    plt.title("max_haus")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "{}_max_haus".format(epoch)))
    plt.clf()
    norm_list = [i/max_x for i in max_haus_list]
    l1 = plt.plot(norm_list, nn_train_15, "ro-", label="nn_train")
    l2 = plt.plot(norm_list, nn_test_15, "r+-", label="nn_test")
    l3 = plt.plot(norm_list, ppr_train, "bo-", label="ppr_train")
    l4 = plt.plot(norm_list, ppr_test, "b+-", label="ppr_test")
    l5 = plt.plot(norm_list, b_train_15, "go-", label="b_train")
    l6 = plt.plot(norm_list, b_test_15, "g+-", label="b_test")
    plt.title("norm_haus")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "{}_norm_haus".format(epoch)))
    plt.clf()

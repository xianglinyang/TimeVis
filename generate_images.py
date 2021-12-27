import torch
import sys
import os
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
from singleVis.visualizer import visualizer

from singleVis.backend import fuzzy_complex, boundary_wise_complex, construct_step_edge_dataset, \
    construct_temporal_edge_dataset, get_attention, construct_temporal_edge_dataset2
import singleVis.config as config
import argparse
parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--content_path', type=str)
parser.add_argument('-d','--dataset', choices=['cifar10', 'mnist', 'fmnist','online'])

args = parser.parse_args()

CONTENT_PATH = args.content_path
DATASET = args.dataset
LEN = config.dataset_config[DATASET]["TRAINING_LEN"]
LAMBDA = config.dataset_config[DATASET]["LAMBDA"]

# define hyperparameters

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCH_NUMS = config.dataset_config[DATASET]["training_config"]["EPOCH_NUM"]
TIME_STEPS = config.dataset_config[DATASET]["training_config"]["TIME_STEPS"]
TEMPORAL_PERSISTENT = config.dataset_config[DATASET]["training_config"]["TEMPORAL_PERSISTENT"]
NUMS = config.dataset_config[DATASET]["training_config"]["NUMS"]    # how many epoch should we go through for one pass
PATIENT = config.dataset_config[DATASET]["training_config"]["PATIENT"]

content_path = CONTENT_PATH
sys.path.append(content_path)

from Model.model import *
net = resnet18()
classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

data_provider = DataProvider(content_path, net, 1, TIME_STEPS, 1, split=-1, device=DEVICE, verbose=1)

model = SingleVisualizationModel(input_dims=512, output_dims=2, units=256)
negative_sample_rate = 5
min_dist = .1
_a, _b = find_ab_params(1.0, min_dist)
umap_loss_fn = UmapLoss(negative_sample_rate, DEVICE, _a, _b, repulsion_strength=1.0)
recon_loss_fn = ReconstructionLoss(beta=1.0)
criterion = SingleVisLoss(umap_loss_fn, recon_loss_fn, lambd=LAMBDA)
optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)

trainer = SingleVisTrainer(model, criterion, optimizer, edge_loader=None, DEVICE=DEVICE)
trainer.load(file_path=os.path.join(data_provider.model_path,"SV.pth"))

########################################################################################################################
# visualization results
########################################################################################################################
classes = list(range(10))
vis = visualizer(data_provider, trainer.model, 200, 10, classes)
save_dir = "./result"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
for i in range(1, TIME_STEPS+1, 1):
    test_data = data_provider.test_representation(i)
    test_labels = data_provider.test_labels(i)
    with open("/home/xianglin/projects/DVI_data/online_learning/target_list.json", "r") as f:
        index = json.load(f)
    test_data = test_data[index]
    test_labels = test_labels[index]
    preds = data_provider.get_pred(i, test_data)
    preds = np.argmax(preds, axis=1)
    vis.savefig_cus(i,test_data, preds, test_labels, path=os.path.join(save_dir, "motivated_{}_{}.png".format(DATASET, i)))
########################################################################################################################
# evaluate
########################################################################################################################
# from singleVis.eval.evaluator import Evaluator
# evaluator = Evaluator(data_provider, trainer)
# evaluator.save_eval(n_neighbors=15, file_name="evaluation")

import torch
import sys
import os
import numpy as np
import argparse
from umap.umap_ import find_ab_params

sys.path.append("/home/xianglin/projects/git_space/SingleVisualization")
from singleVis.SingleVisualizationModel import SingleVisualizationModel
from singleVis.losses import SingleVisLoss, UmapLoss, ReconstructionLoss
from singleVis.trainer import SingleVisTrainer
from singleVis.data import DataProvider
import singleVis.config as config
from scipy import stats

########################################################################################################################
#                                                     LOAD PARAMETERS                                                  #
########################################################################################################################


parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--content_path', type=str)
parser.add_argument('-d','--dataset', choices=['online','cifar10', 'mnist', 'fmnist', 'cifar10_full', 'mnist_full', 'fmnist_full'])
parser.add_argument('-p',"--preprocess", choices=[0,1], default=0)
parser.add_argument('-g',"--gpu_id", type=int, choices=[0,1,2,3], default=2)
parser.add_argument('--stage', type=int)
args = parser.parse_args()

CONTENT_PATH = args.content_path
DATASET = args.dataset
PREPROCESS = args.preprocess
GPU_ID = args.gpu_id
STAGE = args.stage

LEN = config.dataset_config[DATASET]["TRAINING_LEN"]
TEST_LEN = config.dataset_config[DATASET]["TESTING_LEN"]
LAMBDA = config.dataset_config[DATASET]["LAMBDA"]
L_BOUND = config.dataset_config[DATASET]["L_BOUND"]
MAX_HAUSDORFF = config.dataset_config[DATASET]["MAX_HAUSDORFF"]
ALPHA = config.dataset_config[DATASET]["ALPHA"]
BETA = config.dataset_config[DATASET]["BETA"]
INIT_NUM = config.dataset_config[DATASET]["INIT_NUM"]
EPOCH_START = config.dataset_config[DATASET]["EPOCH_START"]
EPOCH_END = config.dataset_config[DATASET]["EPOCH_END"]
EPOCH_PERIOD = config.dataset_config[DATASET]["EPOCH_PERIOD"]
HIDDEN_LAYER = config.dataset_config[DATASET]["HIDDEN_LAYER"]
VIS_MODEL_NAME = config.dataset_config[DATASET]["VIS_MODEL_NAME"]
EVAL_NAME = config.dataset_config[DATASET]["EVAL_NAME"]

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


########################################################################################################################
#                                                    TRAINING SETTING                                                  #
########################################################################################################################
data_provider = DataProvider(content_path, net, EPOCH_START, EPOCH_END, EPOCH_PERIOD, split=-1, device=DEVICE, verbose=1)
if PREPROCESS:
    data_provider.initialize(LEN//10, l_bound=L_BOUND)

model = SingleVisualizationModel(input_dims=512, output_dims=2, units=256, hidden_layer=HIDDEN_LAYER)
negative_sample_rate = 5
min_dist = .1
_a, _b = find_ab_params(1.0, min_dist)
umap_loss_fn = UmapLoss(negative_sample_rate, DEVICE, _a, _b, repulsion_strength=1.0)
recon_loss_fn = ReconstructionLoss(beta=1.0)
criterion = SingleVisLoss(umap_loss_fn, recon_loss_fn, lambd=LAMBDA)

optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)


trainer = SingleVisTrainer(model, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler, edge_loader=None, DEVICE=DEVICE)
trainer.load(file_path=os.path.join(data_provider.model_path,"{}.pth".format(VIS_MODEL_NAME)))

########################################################################################################################
#                                                      CORRELATION                                                     #
########################################################################################################################
EPOCH = (EPOCH_END-EPOCH_START)//EPOCH_PERIOD+1

timeline = np.arange(0, EPOCH, 1)
stage_idxs =  np.array_split(timeline, STAGE)

# train
all_train_repr = np.zeros((EPOCH,LEN,512))
for i in range(EPOCH_START,EPOCH_END + 1, EPOCH_PERIOD):
    index = (i-EPOCH_START)//EPOCH_PERIOD
    all_train_repr[index] = data_provider.train_representation(i)

model = trainer.model
low_repr = np.zeros((EPOCH,LEN,2))
for e in range(EPOCH):
    low_repr[e] = model.encoder(torch.from_numpy(all_train_repr[e]).to(device=DEVICE).float()).detach().cpu().numpy()


epochs = [i for i in range(EPOCH)]
corrs = np.zeros((EPOCH,LEN))
ps = np.zeros((EPOCH,LEN))
tnn = np.zeros((EPOCH,LEN))

for i in range(LEN):
    high_embeddings = all_train_repr[:,i,:].squeeze()
    low_embeddings = low_repr[:,i,:].squeeze()

    for e in epochs:
        selected_stage = stage_idxs[np.where([e in idx for idx in stage_idxs])[0][0]]
        high_dists = np.linalg.norm(high_embeddings - high_embeddings[e], axis=1)
        low_dists = np.linalg.norm(low_embeddings - low_embeddings[e], axis=1)
        corr, p = stats.spearmanr(high_dists[selected_stage], low_dists[selected_stage])
        corrs[e][i] = corr
        ps[e][i] = p 

        # find the index of top k dists
        high_rankings = np.argsort(high_dists)[1:6]
        low_rankings = np.argsort(low_dists)[1:6]
        tnn[e][i] = len(np.intersect1d(high_rankings, low_rankings))

np.save(os.path.join(content_path, "Model", "noB_small_tnn_{}_corrs.npy".format(STAGE)), corrs)
np.save(os.path.join(content_path, "Model", "noB_small_tnn_{}_ps.npy".format(STAGE)), ps)
np.save(os.path.join(content_path, "Model", "noB_small_tnn_{}_5_tnn.npy".format(STAGE)), tnn)

# test
all_test_repr = np.zeros((EPOCH,TEST_LEN,512))
for i in range(EPOCH_START,EPOCH_END + 1, EPOCH_PERIOD):
    index = (i-EPOCH_START)//EPOCH_PERIOD
    all_test_repr[index] = data_provider.test_representation(i)

low_repr = np.zeros((EPOCH,TEST_LEN,2))
for e in range(EPOCH):
    low_repr[e] = model.encoder(torch.from_numpy(all_test_repr[e]).to(device=DEVICE).float()).detach().cpu().numpy()

epochs = [i for i in range(EPOCH)]
corrs = np.zeros((EPOCH,TEST_LEN))
ps = np.zeros((EPOCH,TEST_LEN))
tnn = np.zeros((EPOCH,TEST_LEN))

for i in range(TEST_LEN):
    high_embeddings = all_test_repr[:,i,:].squeeze()
    low_embeddings = low_repr[:,i,:].squeeze()

    for e in epochs:
        selected_stage = stage_idxs[np.where([e in idx for idx in stage_idxs])[0][0]]
        high_dists = np.linalg.norm(high_embeddings - high_embeddings[e], axis=1)
        low_dists = np.linalg.norm(low_embeddings - low_embeddings[e], axis=1)
        corr, p = stats.spearmanr(high_dists[selected_stage], low_dists[selected_stage])
        corrs[e][i] = corr
        ps[e][i] = p 

        # find the index of top k dists
        high_rankings = np.argsort(high_dists)[1:6]
        low_rankings = np.argsort(low_dists)[1:6]
        tnn[e][i] = len(np.intersect1d(high_rankings, low_rankings))

np.save(os.path.join(content_path, "Model", "noB_small_tnn_test_{}_corrs.npy".format(STAGE)), corrs)
np.save(os.path.join(content_path, "Model", "noB_small_tnn_test_{}_ps.npy".format(STAGE)), ps)
np.save(os.path.join(content_path, "Model", "noB_small_tnn_test_{}_5_tnn.npy".format(STAGE)), tnn)

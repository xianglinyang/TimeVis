import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

import edge_dataset
from SingleVisualizationModel import SingleVisualizationModel
from losses import SingleVisLoss
from edge_dataset import DataHandler
from trainer import SingleVisTrainer


# define parameters
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCH_NUMS = 100
NUMS = 10000


model = SingleVisualizationModel()
criterion = SingleVisLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=.1)

# TODO construct spatio-temporal complex and get edges
# dummy input
edge_to = None
edge_from = None
feature_vectors = None
dataset = DataHandler(edge_to, edge_from, feature_vectors)

sampler = WeightedRandomSampler(probs, NUMS, replacement=False)

edge_loader = DataLoader(dataset, batch_size=1000, sampler=sampler)

trainer = SingleVisTrainer(model, criterion, optimizer, edge_loader=edge_loader, DEVICE=DEVICE)
for epoch in EPOCH_NUMS:
    trainer.train_step()
    eval = trainer.eval()
    # early stop, check whether converge or not
trainer.save(name="cifar10_epoch_10")

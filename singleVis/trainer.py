import torch
from torch.utils.data import DataLoader

import edge_dataset
from SingleVisualizationModel import SingleVisualizationModel
from losses import SingleVisLoss
from edge_dataset import DataHandler


# define parameters
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

kwargs = {"shuffle": True}      # loader hyperparameter
loader = DataLoader(dataset, **kwargs)

num_epochs = 100    # a large value before early stopping

for epoch in range(num_epochs):
    all_loss = []
    for data in loader:
        edge_to, edge_from = data
        edge_to.to(DEVICE)
        # ===================forward=====================
        outputs = model(edge_to, edge_from)
        loss = criterion(outputs)
        all_loss.append(loss.item())
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    lr_scheduler.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, sum(all_loss) / len(all_loss)))

# save all parameters...
save_model = {
    "epoch": epoch,
    "loss": sum(all_loss) / len(all_loss),
    "state_dict": model.state_dict(),
    "optimizer": optimizer.state_dict()}
torch.save(save_model, 'singleVisModel.pth')
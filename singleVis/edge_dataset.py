from torch.utils.data import Dataset


# TODO two hierarchy index,
class DataHandler(Dataset):
    def __init__(self, edge_to, edge_from, feature_vector):
        self.edge_to = edge_to
        self.edge_from = edge_from
        self.data = feature_vector

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

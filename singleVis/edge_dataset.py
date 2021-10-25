from torch.utils.data import Dataset
from PIL import Image


# TODO two hierarchy index,
class DataHandler(Dataset):
    def __init__(self, edge_to, edge_from, feature_vector, transform=None):
        self.edge_to = edge_to
        self.edge_from = edge_from
        self.data = feature_vector
        self.transfrom = transform

    def __getitem__(self, item):
        edge_to = self.edge_to[item]
        edge_from = self.edge_from[item]
        edge_to = self.data[edge_to]
        edge_from = self.data[edge_from]
        if self.transfrom is not None:
            # TODO test this
            edge_to = Image.fromarray(edge_to)
            edge_to = self.transfrom(edge_to)
            edge_from = Image.fromarray(edge_from)
            edge_from = self.transfrom(edge_from)
        return edge_to, edge_from

    def __len__(self):
        return len(self.data)

from torch import nn


class SingleVisualizationModel(nn.Module):
    def __init__(self, input_dims, output_dims, units, hidden_layer=3):
        super(SingleVisualizationModel, self).__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.units = units
        self.hidden_layer = hidden_layer
        self._init_autoencoder()
    
    # TODO find the best model architecture
    def _init_autoencoder(self):
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dims, self.units),
            nn.ReLU(True))
        for h in range(self.hidden_layer):
            self.encoder.add_module("{}".format(2*h), nn.Linear(self.units, self.units))
            self.encoder.add_module("{}".format(2*h), nn.ReLU(True))
        self.encoder.add_module("{}".format(2*(self.hidden_layer+1)), nn.Linear(self.units, self.output_dims))

        self.decoder = nn.Sequential(
            nn.Linear(self.output_dims, self.units),
            nn.ReLU(True))
        for h in range(self.hidden_layer):
            self.encoder.add_module("{}".format(2*h), nn.Linear(self.units, self.units))
            self.encoder.add_module("{}".format(2*h), nn.ReLU(True))
        self.decoder.add_module("{}".format(2*(self.hidden_layer+1)), nn.Linear(self.units, self.input_dims))

    def forward(self, edge_to, edge_from):
        outputs = dict()
        embedding_to = self.encoder(edge_to)
        embedding_from = self.encoder(edge_from)
        recon_to = self.decoder(embedding_to)
        recon_from = self.decoder(embedding_from)
        
        outputs["umap"] = (embedding_to, embedding_from)
        outputs["recon"] = (recon_to, recon_from)

        return outputs

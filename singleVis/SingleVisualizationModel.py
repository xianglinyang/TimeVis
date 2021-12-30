from torch import nn


class SingleVisualizationModel(nn.Module):
    def __init__(self, input_dims, output_dims, units):
        super(SingleVisualizationModel, self).__init__()

        # TODO find the best model architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, units),
            nn.ReLU(True),
            nn.Linear(units, units),
            nn.ReLU(True),
            nn.Linear(units, units),
            nn.ReLU(True),
            nn.Linear(units, units),
            nn.ReLU(True),
            nn.Linear(units, output_dims)
        )

        self.decoder = nn.Sequential(
            nn.Linear(output_dims, units),
            nn.ReLU(True),
            nn.Linear(units, units),
            nn.ReLU(True),
            nn.Linear(units, units),
            nn.ReLU(True),
            nn.Linear(units, units),
            nn.ReLU(True),
            nn.Linear(units, input_dims)
        )

    def forward(self, edge_to, edge_from):
        outputs = dict()
        embedding_to = self.encoder(edge_to)
        embedding_from = self.encoder(edge_from)
        recon_to = self.decoder(embedding_to)
        recon_from = self.decoder(embedding_from)
        
        outputs["umap"] = (embedding_to, embedding_from)
        outputs["recon"] = (recon_to, recon_from)

        return outputs

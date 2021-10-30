import torch
import math
import tqdm
import numpy as np


def convert_distance_to_probability(distances, a=1.0, b=1.0):
    return 1.0 / (1.0 + a * torch.pow(distances, 2 * b))


def compute_cross_entropy(
        probabilities_graph, probabilities_distance, EPS=1e-4, repulsion_strength=1.0
):
    """
    Compute cross entropy between low and high probability
    Parameters
    ----------
    probabilities_graph : torch.Tensor
        high dimensional probabilities
    probabilities_distance : torch.Tensor
        low dimensional probabilities
    # probabilities : array
    #     edge weights + zeros
    EPS : float, optional
        offset to to ensure log is taken of a positive number, by default 1e-4
    repulsion_strength : float, optional
        strength of repulsion between negative samples, by default 1.0
    Returns
    -------
    attraction_term: torch.float
        attraction term for cross entropy loss
    repellent_term: torch.float
        repellent term for cross entropy loss
    cross_entropy: torch.float
        cross entropy umap loss
    """
    attraction_term = - probabilities_graph * torch.log(torch.clamp(probabilities_distance, min=EPS, max=1.0))
    repellent_term = (
            -(1.0 - probabilities_graph)
            * torch.log(torch.clamp(1.0 - probabilities_distance, min=EPS, max=1.0))
            * repulsion_strength
    )

    # balance the expected losses between attraction and repel
    CE = attraction_term + repellent_term
    return attraction_term, repellent_term, CE


def batch_run(model, data, output_shape, batch_size=200):
    """batch run, in case memory error"""
    data = data.to(dtype=torch.float)
    input_X = np.zeros([len(data), output_shape])
    n_batches = max(math.ceil(len(data) / batch_size), 1)
    for b in tqdm.tqdm(range(n_batches)):
        r1, r2 = b * batch_size, (b + 1) * batch_size
        inputs = data[r1:r2]
        with torch.no_grad():
            pred = model(inputs).cpu().numpy()
            input_X[r1:r2] = pred.reshape(pred.shape[0], -1)
    return input_X
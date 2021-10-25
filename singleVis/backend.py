"""
backend APIs for Single Visualization model trainer
"""
# import modules
from umap.umap_ import fuzzy_simplicial_set

from pynndescent import NNDescent
from sklearn.utils import check_random_state
from sklearn.neighbors import KDTree
import numpy as np


def get_graph_elements(graph_, n_epochs):
    """
    gets elements of graphs, weights, and number of epochs per edge
    Parameters
    ----------
    graph_ : scipy.sparse.csr.csr_matrix
        umap graph of probabilities
    n_epochs : int
        maximum number of epochs per edge
    Returns
    -------
    graph scipy.sparse.csr.csr_matrix
        umap graph
    epochs_per_sample np.array
        number of epochs to train each sample for
    head np.array
        edge head
    tail np.array
        edge tail
    weight np.array
        edge weight
    n_vertices int
        number of verticies in graph
    """
    ### should we remove redundancies () here??
    # graph_ = remove_redundant_edges(graph_)

    graph = graph_.tocoo()
    # eliminate duplicate entries by summing them together
    graph.sum_duplicates()
    # number of vertices in dataset
    n_vertices = graph.shape[1]
    # get the number of epochs based on the size of the dataset
    if n_epochs is None:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200
    # remove elements with very low probability
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()

    head = graph.row
    tail = graph.col
    weight = graph.data

    return graph, head, tail, weight, n_vertices


def fuzzy_complex(train_data, n_neighbors):
    """
    construct a vietoris-rips complex
    """
    # number of trees in random projection forest
    n_trees = min(64, 5 + int(round((train_data.shape[0]) ** 0.5 / 20.0)))
    # max number of nearest neighbor iters to perform
    n_iters = max(5, int(round(np.log2(train_data.shape[0]))))
    # distance metric
    metric = "euclidean"
    # get nearest neighbors
    nnd = NNDescent(
        train_data,
        n_neighbors=n_neighbors,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=True
    )
    knn_indices, knn_dists = nnd.neighbor_graph
    random_state = check_random_state(None)
    complex, sigmas, rhos = fuzzy_simplicial_set(
        X=train_data,
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=random_state,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
    )
    return complex, sigmas, rhos


def boundary_wise_complex(train_data, border_centers, n_neighbors):
    """
    compute the boundary wise complex
        for each border point, we calculate its k nearest train points
        for each train data, we calculate its k nearest border points
    :param train_data:
    :param border_centers:
    :param n_neighbors:
    :return:
    """
    high_tree = KDTree(border_centers)

    fitting_data = np.concatenate((train_data, border_centers), axis=0)
    knn_dists, knn_indices = high_tree.query(fitting_data, k=n_neighbors)
    knn_indices = knn_indices + len(train_data)

    random_state = check_random_state(None)
    bw_complex, sigmas, rhos = fuzzy_simplicial_set(
        X=fitting_data,
        n_neighbors=n_neighbors,
        metric="euclidean",
        random_state=random_state,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
    )
    return bw_complex, sigmas, rhos


def construct_edge_dataset(X_input, vr_complex, bw_complex):
    """
    construct the mixed edge dataset
        connect border points and train data(both direction)
    :param X_input: tuple (train_data, border_points)
    :param vr_complex: Vietoris-Rips complex
    :param bw_complex: boundary-augmented complex
    :param batch_size: edge dataset batch size
    :return: edge dataset
    """

    train_data, border_centers = X_input
    fitting_data = np.concatenate((train_data, border_centers), axis=0)

    # get data from graph
    _, vr_head, vr_tail, vr_weight, _ = get_graph_elements(vr_complex)
    # get data from graph
    _, bw_head, bw_tail, bw_weight, _ = get_graph_elements(bw_complex)

    head = np.concatenate((vr_head, bw_head), axis=0)
    tail = np.concatenate((vr_tail, bw_tail), axis=0)
    weight = np.concatenate((vr_head, bw_weight), axis=0)

    return head, tail, weight


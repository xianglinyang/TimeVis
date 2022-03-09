"""
backend APIs for Single Visualization model trainer
"""
# import modules
import os
import json
import torch
import time
import math
import numpy as np
import scipy.sparse
from scipy.special import softmax

from umap.umap_ import fuzzy_simplicial_set, compute_membership_strengths
from pynndescent import NNDescent
from sklearn.utils import check_random_state
from sklearn.neighbors import KDTree, NearestNeighbors

from singleVis.utils import hausdorff_dist_cus, jaccard_similarity, knn, hausdorff_dist
from singleVis.kcenter_greedy import kCenterGreedy
from singleVis.intrinsic_dim import IntrinsicDim

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
    # # get the number of epochs based on the size of the dataset
    if n_epochs is None:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200
    # remove elements with very low probability
    graph.data[graph.data < (graph.data.max() / float(n_epochs)) + 1e-3] = 0.0
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
    return complex, sigmas, rhos, knn_indices


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
    return bw_complex, sigmas, rhos, knn_indices


def construct_step_edge_dataset(vr_complex, bw_complex, n_epochs):
    """
    construct the mixed edge dataset for one time step
        connect border points and train data(both direction)
    :param vr_complex: Vietoris-Rips complex
    :param bw_complex: boundary-augmented complex
    :param n_epochs: the number of epoch that we iterate each round
    :return: edge dataset
    """
    # get data from graph
    _, vr_head, vr_tail, vr_weight, _ = get_graph_elements(vr_complex, n_epochs)
    # get data from graph
    _, bw_head, bw_tail, bw_weight, _ = get_graph_elements(bw_complex, 1)

    head = np.concatenate((vr_head, bw_head), axis=0)
    tail = np.concatenate((vr_tail, bw_tail), axis=0)
    weight = np.concatenate((vr_weight, bw_weight), axis=0)

    return head, tail, weight
    # return vr_head, vr_tail, vr_weight


def knn_dists(X, indices, knn_indices):
    data = X[indices][:,None,:]
    knn_data = X[knn_indices]
    knn_dists = np.linalg.norm(data-knn_data, axis=2)
    return knn_dists


def convert_distance_to_probability(distances, a=1.0, b=1.0):
    """convert distance to student-t distribution probability in low-dimensional space"""
    return 1.0 / (1.0 + a * torch.pow(distances, 2 * b))

def get_unit(data, init_num=200, adding_num=100):
    t0 = time.time()
    l = len(data)
    idxs = np.random.choice(np.arange(l), size=init_num, replace=False)
    # _,_ = hausdorff_dist_cus(data, idxs)

    id = IntrinsicDim(data)
    d0 = id.twonn_dimension_fast()
    # d0 = twonn_dimension_fast(data)

    kc = kCenterGreedy(data)
    _ = kc.select_batch_with_budgets(idxs, adding_num)
    c0 = kc.hausdorff()
    t1 = time.time()
    return c0, d0, "{:.1f}".format(t1-t0)


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


def construct_temporal_complex(X, time_step_nums, time_step_idxs_list, persistent, time_steps, knn_indices, sigmas, rhos, k=15):
    """
    construct temporal edges based on same data
    link data to its next epoch
    :param time_step_nums: [(train_num, b_num)]
    :param time_step_idxs_list: list of index to prev train_data
    :param persistent: the length of sliding window
    :param time_steps: the number of time steps we are looking at
    :param knn_indices: (n_vertices*1.1*time_steps, k)
    :return:
    """
    rows = np.zeros(1, dtype=np.int32)
    cols = np.zeros(1, dtype=np.int32)
    vals = np.zeros(1, dtype=np.float32)
    n_all = 0
    time_step_num = list()
    for i in time_step_nums:
        time_step_num.append(n_all)
        n_all = n_all + i[0]
    n_all = 0
    all_step_num = list()
    for i in time_step_nums:
        all_step_num.append(n_all)
        n_all = n_all + i[0] + i[1]
    
    # forward
    for window in range(1, persistent + 1, 1):
        for step in range(0, time_steps - window, 1):
            knn_indices_in = - np.ones((n_all, k))
            knn_dist = np.zeros((n_all, k))

            next_knn = knn_indices[time_step_num[step+window]:time_step_num[step+window] + time_step_nums[step + window][0]]

            # knn_indices_in[all_step_num[step]: all_step_num[step] + time_step_nums[step + window][0]] = next_knn
            increase_idx = all_step_num[step]
            assert len(next_knn) == len(time_step_idxs_list[step+window])
            for i in range(len(time_step_idxs_list[step+window])):
                knn_indices_in[increase_idx + time_step_idxs_list[step+window][i]]=next_knn[i]
            knn_indices_in = knn_indices_in.astype('int')

            indices = np.arange(all_step_num[step], all_step_num[step] + time_step_nums[step][0], 1)[time_step_idxs_list[step+window]]
            knn_dists_t = knn_dists(X, indices, next_knn)

            # knn_dist[all_step_num[step]:all_step_num[step] + time_step_nums[step + window][0]] = knn_dists_t
            assert len(knn_dists_t) == len(time_step_idxs_list[step+window])
            for i in range(len(time_step_idxs_list[step+window])):
                knn_dist[increase_idx + time_step_idxs_list[step+window][i]]=knn_dists_t[i]
            knn_dist = knn_dist.astype('float32')

            rows_t, cols_t, vals_t, _ = compute_membership_strengths(knn_indices_in, knn_dist, sigmas, rhos, return_dists=False)
            idxs = vals_t > 0
            rows = np.concatenate((rows, rows_t[idxs]), axis=0)
            cols = np.concatenate((cols, cols_t[idxs]), axis=0)
            vals = np.concatenate((vals, vals_t[idxs]), axis=0)
    # backward
    for window in range(1, persistent + 1, 1):
        for step in range(time_steps-1, 0 + window, -1):
            knn_indices_in = - np.ones((n_all, k))
            knn_dist = np.zeros((n_all, k))

            prev_knn = knn_indices[time_step_num[step-window]:time_step_num[step-window] + time_step_nums[step-window][0]]

            knn_indices_in[all_step_num[step]: all_step_num[step] + time_step_nums[step][0]] = prev_knn[time_step_idxs_list[step]]
            knn_indices_in = knn_indices_in.astype('int')

            indices = np.arange(all_step_num[step], all_step_num[step] + time_step_nums[step][0], 1)
            knn_dists_t = knn_dists(X, indices, prev_knn[time_step_idxs_list[step]])

            knn_dist[all_step_num[step]:all_step_num[step] + time_step_nums[step][0]] = knn_dists_t
            knn_dist = knn_dist.astype('float32')

            rows_t, cols_t, vals_t, _ = compute_membership_strengths(knn_indices_in, knn_dist, sigmas, rhos, return_dists=False)
            idxs = vals_t > 0
            rows = np.concatenate((rows, rows_t[idxs]), axis=0)
            cols = np.concatenate((cols, cols_t[idxs]), axis=0)
            vals = np.concatenate((vals, vals_t[idxs]), axis=0)
    time_complex = spatio_temporal_simplicial_set(rows=rows, cols=cols, vals=vals, n_vertice=len(X))
    
    return time_complex



def construct_temporal_complex_tnn(X, time_step_nums, time_steps, sigmas, rhos, n_neighbors=5):
    """
    construct temporal edges, connect each sample to its nearest temporal neighbors
    :param X: feature vectors, ((train_num+b_num), feature_dim)
    :param time_step_nums: [(train_num, b_num)]
    :param time_steps: the number of time steps we are looking at
    :param sigma: shape ((train_num+b_num), feature_dim)
    :param rhos: shape ((train_num+b_num), feature_dim)
    :param n_neighbors: the number of temporal neighbors we are looking at
    :return:
    """
    rows = np.zeros(1, dtype=np.int32)
    cols = np.zeros(1, dtype=np.int32)
    vals = np.zeros(1, dtype=np.float32)

    base_idx = 0
    base_idx_list = list()
    for i in time_step_nums:
        base_idx_list.append(base_idx)
        base_idx = base_idx + i[0] + i[1]
    base_idx_list = np.array(base_idx_list, dtype=int)

    valid_idx_list = list()
    for i in range(len(time_step_nums)):
        valid_idx_list.append(base_idx_list[i]+time_step_nums[i][0])
    valid_idx_list = np.array(valid_idx_list, dtype=int)
    
    num = len(X)

    # placeholder for knn_indices and knn_dists
    indices = - np.ones((num, n_neighbors), dtype=int)
    dists = np.zeros((num, n_neighbors), dtype=np.float32)

    for time_step in range(time_steps):
        start_idx = base_idx_list[time_step]
        end_idx = start_idx + time_step_nums[time_step][0]
        # move_positions = [i - start_idx for i in base_idx_list]
        move_positions = base_idx_list - start_idx
        for train_sample_idx in range(start_idx, end_idx + 1, 1):
            # candidate_idxs = [train_sample_idx + i for i in move_positions if train_sample_idx + i < valid_idx]
            candidate_idxs = train_sample_idx + move_positions
            candidate_idxs = candidate_idxs[np.logical_and(candidate_idxs>=base_idx_list, candidate_idxs<valid_idx_list)]
            nn_dist = knn_dists(X, [train_sample_idx], candidate_idxs).squeeze(axis=0)
            # find top k
            order = np.argsort(nn_dist)
            # deal with if len(candidate_idxs)<n_neighbors situation
            top_k_idxs = candidate_idxs[order<n_neighbors]
            top_k_idxs = np.pad(top_k_idxs, (0, n_neighbors-len(top_k_idxs)), 'constant', constant_values=-1).astype('int')
            top_k_dists = nn_dist[order<n_neighbors]
            top_k_dists = np.pad(top_k_dists, (0, n_neighbors-len(top_k_dists)), 'constant', constant_values=0.).astype(np.float32)

            indices[train_sample_idx] = top_k_idxs
            dists[train_sample_idx] = top_k_dists

    rows, cols, vals, _ = compute_membership_strengths(indices, dists, sigmas, rhos, return_dists=False)
    # build time complex
    time_complex = spatio_temporal_simplicial_set(rows=rows, cols=cols, vals=vals, n_vertice=len(X))
    
    return time_complex


# construct spatio-temporal complex and get edges
def construct_spatial_temporal_complex_random(data_provider, init_num, EPOCH_START, EPOCH_END, EPOCH_PERIOD, NUMS, TEMPORAL_PERSISTENT):
    # dummy input
    edge_to = None
    edge_from = None
    sigmas = None
    rhos = None
    weight = None
    probs = None
    feature_vectors = None
    attention = None
    knn_indices = None
    time_step_nums = list()
    time_step_idxs_list = list()

    train_num = data_provider.train_num
    selected_idxs = np.random.choice(np.arange(train_num), size=init_num, replace=False)
    selected_idxs_t = np.array(range(len(selected_idxs)))

    # each time step
    for t in range(EPOCH_START, EPOCH_END+1, EPOCH_PERIOD):
        # load train data and border centers
        train_data = data_provider.train_representation(t).squeeze()
        _, _ = hausdorff_dist_cus(train_data, selected_idxs)

        train_data = train_data[selected_idxs]
        time_step_idxs_list.append(selected_idxs_t.tolist())

        selected_idxs_t = np.random.choice(list(range(len(selected_idxs))), int(0.9*len(selected_idxs)), replace=False)
        selected_idxs = selected_idxs[selected_idxs_t]

        border_centers = data_provider.border_representation(t).squeeze()
        border_centers = border_centers

        complex, sigmas_t1, rhos_t1, knn_idxs_t = fuzzy_complex(train_data, 15)
        bw_complex, sigmas_t2, rhos_t2, _ = boundary_wise_complex(train_data, border_centers, 15)
        edge_to_t, edge_from_t, weight_t = construct_step_edge_dataset(complex, bw_complex, NUMS)
        sigmas_t = np.concatenate((sigmas_t1, sigmas_t2[len(sigmas_t1):]), axis=0)
        rhos_t = np.concatenate((rhos_t1, rhos_t2[len(rhos_t1):]), axis=0)
        fitting_data = np.concatenate((train_data, border_centers), axis=0)
        pred_model = data_provider.prediction_function(t)
        attention_t = get_attention(pred_model, fitting_data, temperature=.01, device=data_provider.DEVICE, verbose=1)
        t_num = len(train_data)
        b_num = len(border_centers)

        if edge_to is None:
            edge_to = edge_to_t
            edge_from = edge_from_t
            weight = weight_t
            probs = weight_t / weight_t.max()
            feature_vectors = fitting_data
            attention = attention_t
            sigmas = sigmas_t
            rhos = rhos_t
            knn_indices = knn_idxs_t
            time_step_nums.append((t_num, b_num))
        else:
            # every round, we need to add len(data) to edge_to(as well as edge_from) index
            increase_idx = len(feature_vectors)
            edge_to = np.concatenate((edge_to, edge_to_t + increase_idx), axis=0)
            edge_from = np.concatenate((edge_from, edge_from_t + increase_idx), axis=0)
            # normalize weight to be in range (0, 1)
            weight = np.concatenate((weight, weight_t), axis=0)
            probs_t = weight_t / weight_t.max()
            probs = np.concatenate((probs, probs_t), axis=0)
            sigmas = np.concatenate((sigmas, sigmas_t), axis=0)
            rhos = np.concatenate((rhos, rhos_t), axis=0)
            feature_vectors = np.concatenate((feature_vectors, fitting_data), axis=0)
            attention = np.concatenate((attention, attention_t), axis=0)
            knn_indices = np.concatenate((knn_indices, knn_idxs_t+increase_idx), axis=0)
            time_step_nums.append((t_num, b_num))

    # boundary points...
    time_steps = (EPOCH_END - EPOCH_START) // EPOCH_PERIOD + 1
    time_complex = construct_temporal_complex(X=feature_vectors,
                                            time_step_nums=time_step_nums,
                                            time_step_idxs_list = time_step_idxs_list,
                                            persistent=TEMPORAL_PERSISTENT,
                                            time_steps=time_steps,
                                            knn_indices=knn_indices,
                                            sigmas=sigmas,
                                            rhos=rhos)
    # normalize for symmetry reason
    _, heads, tails, vals, _ = get_graph_elements(time_complex, n_epochs=NUMS)

    weight = np.concatenate((weight, vals), axis=0)
    probs_t = vals / (vals.max() + 1e-4)
    probs = np.concatenate((probs, probs_t), axis=0)
    edge_to = np.concatenate((edge_to, heads), axis=0)
    edge_from = np.concatenate((edge_from, tails), axis=0)

    return edge_to, edge_from, probs, feature_vectors, attention


# construct spatio-temporal complex and get edges
def construct_spatial_temporal_complex_kc(data_provider, init_num, MAX_HAUSDORFF, ALPHA, BETA, EPOCH_START, EPOCH_END, EPOCH_PERIOD, NUMS, TEMPORAL_PERSISTENT):
    # dummy input
    edge_to = None
    edge_from = None
    sigmas = None
    rhos = None
    weight = None
    probs = None
    feature_vectors = None
    attention = None
    knn_indices = None
    time_step_nums = list()
    time_step_idxs_list = list()

    # train_num = data_provider.train_num
    # selected_idxs = np.random.choice(np.arange(train_num), size=int(train_num * 0.01), replace=False)
    # init_num = 2000
    # target_num = int(math.pow(0.9,(TIME_STEPS))*init_num)

    train_num = data_provider.train_num
    selected_idxs = np.random.choice(np.arange(train_num), size=init_num, replace=False)

    baseline_data = data_provider.train_representation(TIME_STEPS)
    max_x = np.linalg.norm(baseline_data, axis=1).max()
    baseline_data = baseline_data/max_x
    
    c0,d0,_ = get_unit(baseline_data)

    # each time step
    for t in range(EPOCH_END, EPOCH_START - 1, -EPOCH_PERIOD):
        print("=================+++={:d}=+++================".format(t))
        # load train data and border centers
        train_data = data_provider.train_representation(t).squeeze()

        # normalize data by max ||x||_2
        max_x = np.linalg.norm(train_data, axis=1).max()
        train_data = train_data/max_x

        # get normalization parameters for different epochs
        c,d,_ = get_unit(train_data)
        c_c0 = math.pow(c/c0, BETA)
        d_d0 = math.pow(d/d0, ALPHA)
        print("Finish calculating normaling factor")

        kc = kCenterGreedy(train_data)
        _ = kc.select_batch_with_cn(selected_idxs, MAX_HAUSDORFF, c_c0, d_d0, p=0.95)
        selected_idxs = kc.already_selected.astype("int")
        with open("selected_{}.json".format(t), "w") as f:
            json.dump(selected_idxs.tolist(), f)
        print("select {:d} points".format(len(selected_idxs)))

        time_step_idxs_list.insert(0, np.arange(len(selected_idxs)).tolist())

        train_data = data_provider.train_representation(t).squeeze()
        train_data = train_data[selected_idxs]
        border_centers = data_provider.border_representation(t).squeeze()

        t_num = len(selected_idxs)
        b_num = len(border_centers)

        complex, sigmas_t1, rhos_t1, knn_idxs_t = fuzzy_complex(train_data, 15)
        bw_complex, sigmas_t2, rhos_t2, _ = boundary_wise_complex(train_data, border_centers, 15)
        edge_to_t, edge_from_t, weight_t = construct_step_edge_dataset(complex, bw_complex, NUMS)
        sigmas_t = np.concatenate((sigmas_t1, sigmas_t2[len(sigmas_t1):]), axis=0)
        rhos_t = np.concatenate((rhos_t1, rhos_t2[len(rhos_t1):]), axis=0)
        fitting_data = np.concatenate((train_data, border_centers), axis=0)
        pred_model = data_provider.prediction_function(t)
        attention_t = get_attention(pred_model, fitting_data, temperature=.01, device=data_provider.DEVICE, verbose=1)

        if edge_to is None:
            edge_to = edge_to_t
            edge_from = edge_from_t
            weight = weight_t
            probs = weight_t / weight_t.max()
            feature_vectors = fitting_data
            attention = attention_t
            sigmas = sigmas_t
            rhos = rhos_t
            knn_indices = knn_idxs_t
            # npr = npr_t
            time_step_nums.insert(0, (t_num, b_num))
        else:
            # every round, we need to add len(data) to edge_to(as well as edge_from) index
            increase_idx = len(fitting_data)
            edge_to = np.concatenate((edge_to_t, edge_to + increase_idx), axis=0)
            edge_from = np.concatenate((edge_from_t, edge_from + increase_idx), axis=0)
            # normalize weight to be in range (0, 1)
            weight = np.concatenate((weight_t, weight), axis=0)
            probs_t = weight_t / weight_t.max()
            probs = np.concatenate((probs_t, probs), axis=0)
            sigmas = np.concatenate((sigmas_t, sigmas), axis=0)
            rhos = np.concatenate((rhos_t, rhos), axis=0)
            feature_vectors = np.concatenate((fitting_data, feature_vectors), axis=0)
            attention = np.concatenate((attention_t, attention), axis=0)
            knn_indices = np.concatenate((knn_idxs_t, knn_indices+increase_idx), axis=0)
            # npr = np.concatenate((npr_t, npr), axis=0)
            time_step_nums.insert(0, (t_num, b_num))

    # boundary points...
    time_steps = (EPOCH_END - EPOCH_START) // EPOCH_PERIOD + 1
    time_complex = construct_temporal_complex(X=feature_vectors,
                                            time_step_nums=time_step_nums,
                                            time_step_idxs_list = time_step_idxs_list,
                                            persistent=TEMPORAL_PERSISTENT,
                                            time_steps=time_steps,
                                            knn_indices=knn_indices,
                                            sigmas=sigmas,
                                            rhos=rhos)
    # normalize for symmetry reason
    _, heads, tails, vals, _ = get_graph_elements(time_complex, n_epochs=NUMS)

    # increase weight of temporal edges
    # strenthen_neighbor = npr[heads]
    weight = np.concatenate((weight, vals), axis=0)

    probs_t = vals / (vals.max() + 1e-4)
    # probs_t = probs_t*(1+strenthen_neighbor)
    # probs_t = probs_t*TEMPORAL_EDGE_WEIGHT

    probs = np.concatenate((probs, probs_t), axis=0)
    edge_to = np.concatenate((edge_to, heads), axis=0)
    edge_from = np.concatenate((edge_from, tails), axis=0)

    return edge_to, edge_from, probs, feature_vectors, attention


# construct spatio-temporal complex and get edges
def construct_spatial_temporal_complex_kc_tnn(data_provider, init_num, MAX_HAUSDORFF, ALPHA, BETA, EPOCH_START, EPOCH_END, EPOCH_PERIOD, NUMS, TEMPORAL_PERSISTENT):
    # dummy input
    edge_to = None
    edge_from = None
    sigmas = None
    rhos = None
    weight = None
    probs = None
    feature_vectors = None
    attention = None
    knn_indices = None
    time_step_nums = list()
    time_step_idxs_list = list()

    # train_num = data_provider.train_num
    # selected_idxs = np.random.choice(np.arange(train_num), size=int(train_num * 0.01), replace=False)
    # init_num = 2000
    # target_num = int(math.pow(0.9,(TIME_STEPS))*init_num)

    train_num = data_provider.train_num
    selected_idxs = np.random.choice(np.arange(train_num), size=init_num, replace=False)

    baseline_data = data_provider.train_representation(EPOCH_END)
    max_x = np.linalg.norm(baseline_data, axis=1).max()
    baseline_data = baseline_data/max_x
    
    c0,d0,_ = get_unit(baseline_data)

    # each time step
    for t in range(EPOCH_END, EPOCH_START - 1, -EPOCH_PERIOD):
        print("=================+++={:d}=+++================".format(t))
        # load train data and border centers
        train_data = data_provider.train_representation(t).squeeze()

        # select highly used border centers...
        border_centers = data_provider.border_representation(t).squeeze()
        neigh = NearestNeighbors(n_neighbors=15, radius=0.4)
        neigh.fit(border_centers)
        high_ind = neigh.kneighbors(train_data, 15, return_distance=False)
        selected_borders = np.bincount(high_ind.reshape(-1), minlength=len(border_centers))>100
        border_centers = border_centers[selected_borders]

        # normalize data by max ||x||_2
        max_x = np.linalg.norm(train_data, axis=1).max()
        train_data = train_data/max_x

        # get normalization parameters for different epochs
        c,d,_ = get_unit(train_data)
        c_c0 = math.pow(c/c0, BETA)
        d_d0 = math.pow(d/d0, ALPHA)
        print("Finish calculating normaling factor")

        kc = kCenterGreedy(train_data)
        _ = kc.select_batch_with_cn(selected_idxs, MAX_HAUSDORFF, c_c0, d_d0/2, p=0.95)
        selected_idxs = kc.already_selected.astype("int")

        save_dir = os.path.join(data_provider.content_path, "selected_idxs")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        with open(os.path.join(save_dir,"selected_{}.json".format(t)), "w") as f:
            json.dump(selected_idxs.tolist(), f)
        print("select {:d} points".format(len(selected_idxs)))

        time_step_idxs_list.insert(0, np.arange(len(selected_idxs)).tolist())

        train_data = data_provider.train_representation(t).squeeze()
        train_data = train_data[selected_idxs]
        

        t_num = len(selected_idxs)
        b_num = len(border_centers)

        complex, sigmas_t1, rhos_t1, knn_idxs_t = fuzzy_complex(train_data, 15)
        bw_complex, sigmas_t2, rhos_t2, _ = boundary_wise_complex(train_data, border_centers, 15)
        edge_to_t, edge_from_t, weight_t = construct_step_edge_dataset(complex, bw_complex, NUMS)
        sigmas_t = np.concatenate((sigmas_t1, sigmas_t2[len(sigmas_t1):]), axis=0)
        rhos_t = np.concatenate((rhos_t1, rhos_t2[len(rhos_t1):]), axis=0)
        fitting_data = np.concatenate((train_data, border_centers), axis=0)
        pred_model = data_provider.prediction_function(t)
        attention_t = get_attention(pred_model, fitting_data, temperature=.01, device=data_provider.DEVICE, verbose=1)

        if edge_to is None:
            edge_to = edge_to_t
            edge_from = edge_from_t
            weight = weight_t
            probs = weight_t / weight_t.max()
            feature_vectors = fitting_data
            attention = attention_t
            sigmas = sigmas_t
            rhos = rhos_t
            knn_indices = knn_idxs_t
            # npr = npr_t
            time_step_nums.insert(0, (t_num, b_num))
        else:
            # every round, we need to add len(data) to edge_to(as well as edge_from) index
            increase_idx = len(fitting_data)
            edge_to = np.concatenate((edge_to_t, edge_to + increase_idx), axis=0)
            edge_from = np.concatenate((edge_from_t, edge_from + increase_idx), axis=0)
            # normalize weight to be in range (0, 1)
            weight = np.concatenate((weight_t, weight), axis=0)
            probs_t = weight_t / weight_t.max()
            probs = np.concatenate((probs_t, probs), axis=0)
            sigmas = np.concatenate((sigmas_t, sigmas), axis=0)
            rhos = np.concatenate((rhos_t, rhos), axis=0)
            feature_vectors = np.concatenate((fitting_data, feature_vectors), axis=0)
            attention = np.concatenate((attention_t, attention), axis=0)
            knn_indices = np.concatenate((knn_idxs_t, knn_indices+increase_idx), axis=0)
            # npr = np.concatenate((npr_t, npr), axis=0)
            time_step_nums.insert(0, (t_num, b_num))

    # boundary points...
    time_steps = (EPOCH_END - EPOCH_START) // EPOCH_PERIOD + 1
    time_complex = construct_temporal_complex_tnn(X=feature_vectors,
                                                time_step_nums=time_step_nums,
                                                time_steps=time_steps,
                                                sigmas=sigmas,
                                                rhos=rhos)
    # normalize for symmetry reason
    _, heads, tails, vals, _ = get_graph_elements(time_complex, n_epochs=20*NUMS)

    # increase weight of temporal edges
    # strenthen_neighbor = npr[heads]
    weight = np.concatenate((weight, vals), axis=0)

    probs_t = vals / (vals.max() + 1e-4)
    # probs_t = probs_t*(1+strenthen_neighbor)
    # probs_t = probs_t*TEMPORAL_EDGE_WEIGHT

    probs = np.concatenate((probs, probs_t), axis=0)
    edge_to = np.concatenate((edge_to, heads), axis=0)
    edge_from = np.concatenate((edge_from, tails), axis=0)

    return edge_to, edge_from, probs, feature_vectors, attention


def spatio_temporal_simplicial_set(
        rows,
        cols,
        vals,
        n_vertice,
        set_op_mix_ratio=1.0,
        apply_set_operations=True):
    """
    Given the edges and edge weights, compute the simplicial set
    (here represented as a fuzzy graph in the form of a sparse matrix)
    associated to the data.
    This is done by locally approximating geodesic distance at each point,
    creating a fuzzy simplicial set for each such point,
    and then combining all the local fuzzy simplicial sets into a global one via a fuzzy union.
    :param rows: index list of edge_to
    :param cols: index list of edge_from
    :param vals: list of edge weights
    :param n_vertice: int, the number of vertices
    :param set_op_mix_ratio: float (optional, default 1.0)
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.
    :param local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.
    :param apply_set_operations:
    :return: coo_matrix
        A fuzzy simplicial set represented as a sparse matrix. The (i,
        j) entry of the matrix represents the membership strength of the
        1-simplex between the ith and jth sample points.
    """
    result = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(n_vertice, n_vertice)
    )
    result.eliminate_zeros()

    if apply_set_operations:
        transpose = result.transpose()
        prod_matrix = result.multiply(transpose)
        result = (
                set_op_mix_ratio * (result + transpose - prod_matrix)
                + (1.0 - set_op_mix_ratio) * prod_matrix
        )
    result.eliminate_zeros()
    return result


def find_neighbor_preserving_rate(prev_data, train_data, n_neighbors):
    """
    neighbor preserving rate, (0, 1)
    :param prev_data: ndarray, shape(N,2) low dimensional embedding from last epoch
    :param train_data: ndarray, shape(N,2) low dimensional embedding from current epoch
    :param n_neighbors:
    :return alpha: ndarray, shape (N,)
    """
    if prev_data is None:
        return np.zeros(len(train_data))
    # number of trees in random projection forest
    n_trees = min(64, 5 + int(round((train_data.shape[0]) ** 0.5 / 20.0)))
    # max number of nearest neighbor iters to perform
    n_iters = max(5, int(round(np.log2(train_data.shape[0]))))
    # distance metric
    metric = "euclidean"

    from pynndescent import NNDescent
    # get nearest neighbors
    nnd = NNDescent(
        train_data,
        n_neighbors=n_neighbors,
        metric="euclidean",
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=False
    )
    train_indices, _ = nnd.neighbor_graph
    prev_nnd = NNDescent(
        prev_data,
        n_neighbors=n_neighbors,
        metric="euclidean",
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=False
    )
    prev_indices, _ = prev_nnd.neighbor_graph
    temporal_pres = np.zeros(len(train_data))
    for i in range(len(train_indices)):
        pres = np.intersect1d(train_indices[i], prev_indices[i])
        temporal_pres[i] = len(pres) / float(n_neighbors)
    return temporal_pres


def get_attention(model, data, device, temperature=.01, verbose=1):
    t0 = time.time()
    grad_list = []

    for i in range(len(data)):
        b = torch.from_numpy(data[i:i + 1]).to(device=device, dtype=torch.float)
        b.requires_grad = True
        out = model(b)
        top1 = torch.argsort(out)[0][-1]
        out[0][top1].backward()
        grad_list.append(b.grad.data.detach().cpu().numpy())
    grad_list2 = []

    for i in range(len(data)):
        b = torch.from_numpy(data[i:i + 1]).to(device=device, dtype=torch.float)
        b.requires_grad = True
        out = model(b)
        top2 = torch.argsort(out)[0][-2]
        out[0][top2].backward()
        grad_list2.append(b.grad.data.detach().cpu().numpy())
    t1 = time.time()
    grad1 = np.array(grad_list)
    grad2 = np.array(grad_list2)
    grad1 = grad1.squeeze(axis=1)
    grad2 = grad2.squeeze(axis=1)
    grad = np.abs(grad1) + np.abs(grad2)
    grad = softmax(grad/temperature, axis=1)
    t2 = time.time()
    if verbose:
        print("Gradients calculation: {:.2f} seconds\tsoftmax with temperature: {:.2f} seconds".format(round(t1-t0), round(t2-t1)))
    return grad

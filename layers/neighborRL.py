from typing import Any, Dict
import numpy as np
import torch
from torch.functional import Tensor
import math
import os


def pre_node_dist(multi_r_data, features, save_path=None):
    """This is used to culculate the similarity between node and 
    its neighbors in advance in order to avoid the repetitive computation.

    Args:
        multi_r_data ([type]): [description]
        features ([type]): [description]
        save_path ([type], optional): [description]. Defaults to None.
    """

    relation_config: Dict[str, Dict[int, Any]] = {}
    for relation_id, r_data in enumerate(multi_r_data):
        node_config: Dict[int, Any] = {}
        r_data: Tensor
        unique_nodes = r_data[1].unique()
        num_nodes = unique_nodes.size(0)
        for node in range(num_nodes):
            # get neighbors' index
            neighbors_idx = torch.where(r_data[1]==node)[0]
            # get neighbors
            neighbors = r_data[0, neighbors_idx]
            num_neighbors = neighbors.size(0)
            neighbors_features = features[neighbors, :]
            target_features = features[node, :]
            # calculate euclid distance with broadcast
            dist: Tensor = torch.norm(neighbors_features - target_features, p=2, dim=1)
            # smaller is better and we use 'top p' in our paper 
            # (threshold * num_neighbors) see RL_neighbor_filter for details
            sorted_neighbors, sorted_index = dist.sort(descending=False)
            node_config[node] = {'neighbors_idx': neighbors_idx, 
                                 'sorted_neighbors': sorted_neighbors, 
                                 'sorted_index': sorted_index, 
                                 'num_neighbors': num_neighbors}
        relation_config['relation_%d' % relation_id] = node_config

    if save_path is not None:
        save_path = os.path.join(save_path, 'relation_config.npy')
        # print(save_path)
        np.save(save_path, relation_config)


def RL_neighbor_filter(multi_r_data, RL_thresholds, load_path):

    load_path = os.path.join(load_path, 'relation_config.npy')
    relation_config = np.load(load_path, allow_pickle=True)
    relation_config = relation_config.tolist()
    relations = list(relation_config.keys())
    multi_remain_data = []

    for i in range(len(relations)):
        edge_index: Tensor = multi_r_data[i]
        unique_nodes = edge_index[1].unique()
        num_nodes = unique_nodes.size(0)
        remain_node_index = torch.tensor([])
        for node in range(num_nodes):
            # extract config
            neighbors_idx = relation_config[relations[i]][node]['neighbors_idx']
            num_neighbors = relation_config[relations[i]][node]['num_neighbors']
            sorted_neighbors = relation_config[relations[i]][node]['sorted_neighbors']
            sorted_index = relation_config[relations[i]][node]['sorted_index']

            if num_neighbors <= 5: 
                remain_node_index = torch.cat((remain_node_index, neighbors_idx))
                continue # add limitations

            threshold = float(RL_thresholds[i])

            num_kept_neighbors = math.ceil(num_neighbors * threshold) + 1
            filtered_neighbors_idx = neighbors_idx[sorted_index[:num_kept_neighbors]]
            remain_node_index = torch.cat((remain_node_index, filtered_neighbors_idx))

        remain_node_index = remain_node_index.type('torch.LongTensor')
        # print(remain_node_index)
        edge_index = edge_index[:, remain_node_index]
        multi_remain_data.append(edge_index)
    
    return multi_remain_data


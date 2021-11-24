from typing import Any, Dict
import numpy as np
import torch
from torch.functional import Tensor
import math
import os
import time
import gc

def RL_neighbor_filter_full(multi_r_data, RL_thresholds, features, save_path=None):

    multi_remain_data = []
    multi_r_score = []

    for i, r_data in enumerate(multi_r_data):
        r_data: Tensor
        unique_nodes = r_data[1].unique()
        num_nodes = unique_nodes.size(0)
        remain_node_index = torch.tensor([])
        node_scores = []
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
            # => (threshold * num_neighbors)
            # see RL_neighbor_filter for details
            sorted_neighbors, sorted_index = dist.sort(descending=False)
            
            if num_neighbors <= 5: 
                remain_node_index = torch.cat((remain_node_index, neighbors_idx))
                continue # add limitations

            threshold = float(RL_thresholds[i])

            num_kept_neighbors = math.ceil(num_neighbors * threshold) + 1
            filtered_neighbors_idx = neighbors_idx[sorted_index[:num_kept_neighbors]]
            remain_node_index = torch.cat((remain_node_index, filtered_neighbors_idx))

            filtered_neighbors_scores = sorted_neighbors[:num_kept_neighbors].mean()
            node_scores.append(filtered_neighbors_scores)

        remain_node_index = remain_node_index.type('torch.LongTensor')
        edge_index = r_data[:, remain_node_index]
        multi_remain_data.append(edge_index)

        node_scores = torch.FloatTensor(node_scores) # from list
        avg_node_scores = node_scores.sum(dim=1) / num_nodes
        multi_r_score.append(avg_node_scores)

    return multi_remain_data, multi_r_score

def multi_forward_agg(args, foward_args, iter_epoch):

    # args prepare
    model, homo_data, all_num_samples, num_dim, sampler, multi_r_data, filtered_multi_r_data, device, RL_thresholds = foward_args

    if filtered_multi_r_data is None:
        filtered_multi_r_data = multi_r_data
    
    extract_features = torch.FloatTensor([])

    num_batches = int(all_num_samples / args.batch_size) + 1
    
    # all mask are then splited into mini-batch in order
    all_mask = torch.arange(0, num_dim, dtype=torch.long)

    # multiple forward with RL training
    for _ in range(iter_epoch):

        # batch training
        for batch in range(num_batches):
            start_batch = time.time()

            # split batch
            i_start = args.batch_size * batch
            i_end = min((batch + 1) * args.batch_size, all_num_samples)
            batch_nodes = all_mask[i_start:i_end]
            batch_labels = homo_data.y[batch_nodes]

            # sampling neighbors of batch nodes
            adjs, n_ids = sampler.sample(filtered_multi_r_data, node_idx=batch_nodes, sizes=[-1, -1], batch_size=args.batch_size)

            pred = model(homo_data.x, adjs, n_ids, device, RL_thresholds)

            extract_features = torch.cat((extract_features, pred.cpu().detach()), dim=0)

            del pred

        # RL trainig
        filtered_multi_r_data, multi_r_scores = RL_neighbor_filter_full(filtered_multi_r_data, RL_thresholds, extract_features)
        # return new RL thresholds

    return RL_thresholds
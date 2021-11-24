import torch
import torch.nn as nn
from torch.functional import Tensor
import time

from layers.layer import Inter_AGG, Intra_AGG

class MarGNN(nn.Module):
    def __init__(self, GNN_args, num_relations, inter_opt, is_shared=False):
        super(MarGNN, self).__init__()

        self.num_relations = num_relations
        self.inter_opt = inter_opt
        self.is_shared = is_shared
        if not self.is_shared:
            self.intra_aggs = torch.nn.ModuleList([Intra_AGG(GNN_args) for _ in range(self.num_relations)])
        else:
            self.intra_aggs = Intra_AGG(GNN_args) # shared parameters
        
        if self.inter_opt == 'cat_w_avg_mlp' or 'cat_wo_avg_mlp':
            in_dim, hid_dim, out_dim, heads = GNN_args
            mlp_args = self.num_relations * out_dim, out_dim
            self.inter_agg = Inter_AGG(mlp_args)
        else:
            self.inter_agg = Inter_AGG()

    def forward(self, x, adjs, n_ids, device, RL_thresholds):

        # RL_threshold: tensor([[.5], [.5], [.5]])
        if RL_thresholds is None:
            RL_thresholds = torch.FloatTensor([[1.], [1.], [1.]])
        if not isinstance(RL_thresholds, Tensor):
            RL_thresholds = torch.FloatTensor(RL_thresholds)

        RL_thresholds = RL_thresholds.to(device)

        features = []
        for i in range(self.num_relations):
            if not self.is_shared:
                # print('Intra Aggregation of relation %d' % i)
                features.append(self.intra_aggs[i](x[n_ids[i]], adjs[i], device))
            else:
                # shared parameters.
                # print('Shared Intra Aggregation...')
                features.append(self.intra_aggs(x[n_ids[i]], adjs[i], device))

        features = torch.stack(features, dim=0)
        
        features = self.inter_agg(features, RL_thresholds, self.inter_opt) 

        return features
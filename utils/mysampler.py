import copy
from typing import List, Optional, Tuple, NamedTuple, Union, Callable
from scipy import sparse
import random

import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.loader import NeighborSampler, RandomNodeSampler

# def sample(multi_relational_edge_index: List[Tensor], batch, sizes):
#     if not isinstance(batch, Tensor):
#         batch = torch.tensor(batch)
#     batch_size: int = len(batch)
#     outs = []

#     for edge_index in multi_relational_edge_index:
#         num_nodes = int(edge_index.max()) + 1
#         adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], 
#                              value=None,
#                              sparse_sizes=(num_nodes, num_nodes)).t()
#         adj_t.storage.rowptr()

#         adjs = []
#         n_id = batch
#         for size in sizes:
#             adj_t, n_id = adj_t.sample_adj(n_id, size, replace=False)
#             size = adj_t.sparse_sizes()[::-1]
#             row, col, _ = adj_t.coo()
#             edge_index = torch.stack([col, row], dim=0)
#             adjs.append(EdgeIndex(edge_index, size))

#         adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
#         # out = (batch_size, n_id, adjs)
#         out = adjs
#         outs.append(out)

#     return outs

class MySampler(object):

    def __init__(self, sampler) -> None:
        super().__init__()

        self.sampler = sampler

    def sample(self, multi_relational_edge_index: List[Tensor], node_idx, sizes, batch_size):
        
        if self.sampler == 'RL_sampler':
            return self._RL_sample(multi_relational_edge_index, node_idx, sizes, batch_size)
        elif self.sampler == 'random_sampler':
            return self._random_sample(multi_relational_edge_index, node_idx, batch_size)
        elif self.sampler == 'const_sampler':
            return self._const_sample(multi_relational_edge_index, node_idx, batch_size)

    def _RL_sample(self, multi_relational_edge_index: List[Tensor], node_idx, sizes, batch_size):

        outs = []
        all_n_ids = []
        for id, edge_index in enumerate(multi_relational_edge_index):
            loader = NeighborSampler(edge_index=edge_index, 
                                     sizes=sizes, 
                                     node_idx=node_idx,
                                     return_e_id=False,
                                     batch_size=batch_size,
                                     num_workers=0)
            for id, (_, n_ids, adjs) in enumerate(loader):
                # print(adjs)
                outs.append(adjs)
                all_n_ids.append(n_ids)

            # print(id)
            assert id == 0

        return outs, all_n_ids

    def _random_sample(self, multi_relational_edge_index: List[Tensor], node_idx, batch_size):

        outs = []
        all_n_ids = []

        sizes = [random.randint(10, 100), random.randint(10, 50)]

        for edge_index in multi_relational_edge_index:

            loader = NeighborSampler(edge_index=edge_index, 
                                    sizes=sizes, 
                                    node_idx=node_idx,
                                    return_e_id=False,
                                    batch_size=batch_size,
                                    num_workers=0)
            for id, (_, n_ids, adjs) in enumerate(loader):
                # print(adjs)
                outs.append(adjs)
                all_n_ids.append(n_ids)

            # print(id)
            assert id == 0

        return outs, all_n_ids

    def _const_sample(self, multi_relational_edge_index: List[Tensor], node_idx, batch_size):

        outs = []
        all_n_ids = []

        sizes = [25, 15]

        for edge_index in multi_relational_edge_index:

            loader = NeighborSampler(edge_index=edge_index, 
                                    sizes=sizes, 
                                    node_idx=node_idx,
                                    return_e_id=False,
                                    batch_size=batch_size,
                                    num_workers=0)
            for id, (_, n_ids, adjs) in enumerate(loader):
                # print(adjs)
                outs.append(adjs)
                all_n_ids.append(n_ids)

            # print(id)
            assert id == 0

        return outs, all_n_ids

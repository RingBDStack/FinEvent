from scipy.sparse.coo import coo_matrix
from scipy.sparse.csr import csr_matrix
import torch
import numpy as np
from scipy import sparse
from torch_geometric.data import Data
import os

from torch_sparse.tensor import SparseTensor

from .utils import generateMasks, gen_offline_masks

def sparse_trans(datapath = 'incremental_0808/0/s_m_tid_userid_tid.npz'):
    relation = sparse.load_npz(datapath)
    all_edge_index = torch.tensor([], dtype=int)
    for node in range(relation.shape[0]):
        neighbor = torch.IntTensor(relation[node].toarray()).squeeze()
        # del self_loop in advance
        neighbor[node] = 0
        neighbor_idx = neighbor.nonzero()
        neighbor_sum = neighbor_idx.size(0)
        loop = torch.tensor(node).repeat(neighbor_sum, 1)
        edge_index_i_j = torch.cat((loop, neighbor_idx), dim=1).t()
        # edge_index_j_i = torch.cat((neighbor_idx, loop), dim=1).t()
        self_loop = torch.tensor([[node],[node]])
        all_edge_index = torch.cat((all_edge_index, edge_index_i_j, self_loop), dim=1)
        del neighbor, neighbor_idx, loop, self_loop, edge_index_i_j
    return all_edge_index

def coo_trans(datapath = 'incremental_0808/0/s_m_tid_userid_tid.npz'):
    relation:csr_matrix = sparse.load_npz(datapath)
    relation:coo_matrix = relation.tocoo()
    sparse_edge_index = torch.LongTensor([relation.row, relation.col])
    return sparse_edge_index

def create_dataset(loadpath, relation, mode):
    features = np.load(os.path.join(loadpath, str(mode[1]), 'features.npy'))
    features = torch.FloatTensor(features)
    print('features loaded')
    labels = np.load(os.path.join(loadpath, str(mode[1]), 'labels.npy'))
    print('labels loaded')
    labels = torch.LongTensor(labels)
    relation_edge_index = coo_trans(os.path.join(loadpath, str(mode[1]), 's_m_tid_%s_tid.npz' % relation))
    print('edge index loaded')
    data = Data(x=features, edge_index=relation_edge_index, y=labels)
    data_split = np.load(os.path.join(loadpath, 'data_split.npy'))
    train_i, i = mode[0], mode[1]
    if train_i == i:
        data.train_mask, data.val_mask = generateMasks(len(labels), data_split, train_i, i)
    else:
        data.test_mask = generateMasks(len(labels), data_split, train_i, i)

    return data

def create_homodataset(loadpath, mode, valid_percent=0.2):
    features = np.load(os.path.join(loadpath, str(mode[1]), 'features.npy'))
    features = torch.FloatTensor(features)
    print('features loaded')
    labels = np.load(os.path.join(loadpath, str(mode[1]), 'labels.npy'))
    print('labels loaded')
    labels = torch.LongTensor(labels)
    # relation_edge_index = sparse_trans(os.path.join(loadpath, str(mode[1]), 's_bool_A_tid_tid.npz'))
    # print('edge index loaded')
    data = Data(x=features, edge_index=None, y=labels)
    data_split = np.load(os.path.join(loadpath, 'data_split.npy'))
    train_i, i = mode[0], mode[1]
    if train_i == i:
        data.train_mask, data.val_mask = generateMasks(len(labels), data_split, train_i, i, valid_percent)
    else:
        data.test_mask = generateMasks(len(labels), data_split, train_i, i)

    return data

def create_offline_homodataset(loadpath, mode):
    features = np.load(os.path.join(loadpath, str(mode[1]), 'features.npy'))
    features = torch.FloatTensor(features)
    print('features loaded')
    labels = np.load(os.path.join(loadpath, str(mode[1]), 'labels.npy'))
    print('labels loaded')
    labels = torch.LongTensor(labels)
    # relation_edge_index = sparse_trans(os.path.join(loadpath, str(mode[1]), 's_bool_A_tid_tid.npz'))
    # print('edge index loaded')
    data = Data(x=features, edge_index=None, y=labels)
    data.train_mask, data.val_mask, data.test_mask = gen_offline_masks(len(labels))

    return data

# def create_multi_relational_graph(loadpath, relations, mode):
#     features = np.load(os.path.join(loadpath, str(mode[1]), 'features.npy'))
#     features = torch.FloatTensor(features)
#     print('features loaded')
#     labels = np.load(os.path.join(loadpath, str(mode[1]), 'labels.npy'))
#     print('labels loaded')
#     labels = torch.LongTensor(labels)
#     multi_relation_edge_index = [coo_trans(os.path.join(loadpath, str(mode[1]), 's_m_tid_%s_tid.npz' % relation)) for relation in relations]
#     print('edge index loaded')
#     data = [Data(x=features, edge_index=relation_edge_index, y=labels) for relation_edge_index in multi_relation_edge_index]
#     data_split = np.load(os.path.join(loadpath, 'data_split.npy'))
#     train_i, i = mode
#     if train_i == i:
#         train_mask, val_mask = generateMasks(len(labels), data_split, train_i, i)
#         for d in data:
#             d.train_mask, d.val_mask = train_mask, val_mask
#     else:
#         test_mask = generateMasks(len(labels), data_split, train_i, i)
#         for d in data:
#             d.test_mask = test_mask

#     return data

def create_multi_relational_graph(loadpath, relations, mode):

    # multi_relation_edge_index = [sparse_trans(os.path.join(loadpath, str(mode[1]), 's_m_tid_%s_tid.npz' % relation)) for relation in relations]
    multi_relation_edge_index = [torch.load(loadpath + '/' + str(mode[1]) + '/edge_index_%s.pt' % relation) for relation in relations]
    print('sparse trans...')
    print('edge index loaded')

    return multi_relation_edge_index
    
def save_multi_relational_graph(loadpath, relations, mode):

    for relation in relations:
        relation_edge_index = sparse_trans(os.path.join(loadpath, str(mode[1]), 's_m_tid_%s_tid.npz' % relation))
        torch.save(relation_edge_index, loadpath + '/' + str(mode[1]) + '/edge_index_%s.pt' % relation)
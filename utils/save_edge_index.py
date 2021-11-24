import torch
from gen_dataset import save_multi_relational_graph, sparse_trans

# data_path = 'incremental_cross_English_68841'
data_path = 'offline_1115'
relation_ids = ['entity', 'userid', 'word']

# for i in range(22):
#     save_multi_relational_graph(data_path, relation_ids, [0, i])
#     print('edge index saved')

save_multi_relational_graph(data_path, relation_ids, [0, 0])
print('edge index saved')
import torch
from gen_dataset import save_multi_relational_graph

data_path = 'incremental_cross_English_68841'

relation_ids = ['entity', 'userid', 'word']

for i in range(22):
    save_multi_relational_graph(data_path, relation_ids, [0, i])
    print('edge index saved')

print('all edge index saved')
"""
This file splits the Twitter dataset into 21 message blocks (please see Section 4.3 of the paper for more details), 
use the message blocks to construct heterogeneous social graphs (please see Figure 1(a) and Section 3.2 of the paper for more details) 
and maps them into homogeneous message graphs (Figure 1(c)).

Note that:
# 1) We adopt the Latest Message Strategy (which is the most efficient and gives the strongest performance. See Section 4.4 of the paper for more details) here, 
# as a consequence, each message graph only contains the messages of the date and all previous messages are removed from the graph;
# To switch to the All Message Strategy or the Relevant Message Strategy, replace 'G = construct_graph_from_df(incr_df)' with 'G = construct_graph_from_df(incr_df, G)' inside construct_incremental_dataset_0922().
# 2) For test purpose, when calling construct_incremental_dataset_0922(), set test=True, and the message blocks, as well as the resulted message graphs each will contain 100 messages.
# To use all the messages, set test=False, and the number of messages in the message blocks will follow Table. 4 of the paper.
"""
import os
from time import time
import dgl
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse


# construct a heterogeneous social graph using tweet ids, user ids, entities and rare (sampled) words (4 modalities)
# if G is not None then insert new nodes to G
def construct_graph_from_df(df, G=None):
    if G is None:
        G = nx.Graph()
    for _, row in df.iterrows():
        tid = 't_' + str(row['tweet_id'])
        G.add_node(tid)
        G.nodes[tid]['tweet_id'] = True  # right-hand side value is irrelevant for the lookup

        user_ids = row['user_mentions']
        user_ids.append(row['user_id'])
        user_ids = ['u_' + str(each) for each in user_ids]
        # print(user_ids)
        G.add_nodes_from(user_ids)
        for each in user_ids:
            G.nodes[each]['user_id'] = True

        entities = row['entities']
        # entities = ['e_' + each for each in entities]
        # print(entities)
        G.add_nodes_from(entities)
        for each in entities:
            G.nodes[each]['entity'] = True

        words = row['sampled_words']
        words = ['w_' + each for each in words]
        # print(words)
        G.add_nodes_from(words)
        for each in words:
            G.nodes[each]['word'] = True

        edges = []
        edges += [(tid, each) for each in user_ids]
        edges += [(tid, each) for each in entities]
        edges += [(tid, each) for each in words]
        G.add_edges_from(edges)

    return G


# convert a heterogeneous social graph G to a homogeneous message graph following eq. 1 of the paper, 
# and store the sparse binary adjacency matrix of the homogeneous message graph.
def to_dgl_graph_v3(G, save_path=None):
    message = ''
    print('Start converting heterogeneous networkx graph to homogeneous dgl graph.')
    message += 'Start converting heterogeneous networkx graph to homogeneous dgl graph.\n'
    all_start = time()

    print('\tGetting a list of all nodes ...')
    message += '\tGetting a list of all nodes ...\n'
    start = time()
    all_nodes = list(G.nodes)
    mins = (time() - start) / 60
    print('\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # print('All nodes: ', all_nodes)
    # print('Total number of nodes: ', len(all_nodes))

    print('\tGetting adjacency matrix ...')
    message += '\tGetting adjacency matrix ...\n'
    start = time()
    A = nx.to_numpy_matrix(G)  # Returns the graph adjacency matrix as a NumPy matrix.
    mins = (time() - start) / 60
    print('\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # compute commuting matrices
    print('\tGetting lists of nodes of various types ...')
    message += '\tGetting lists of nodes of various types ...\n'
    start = time()
    tid_nodes = list(nx.get_node_attributes(G, 'tweet_id').keys())
    userid_nodes = list(nx.get_node_attributes(G, 'user_id').keys())
    word_nodes = list(nx.get_node_attributes(G, 'word').keys())
    entity_nodes = list(nx.get_node_attributes(G, 'entity').keys())
    del G
    mins = (time() - start) / 60
    print('\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\tConverting node lists to index lists ...')
    message += '\tConverting node lists to index lists ...\n'
    start = time()
    #  find the index of target nodes in the list of all_nodes
    indices_tid = [all_nodes.index(x) for x in tid_nodes]
    indices_userid = [all_nodes.index(x) for x in userid_nodes]
    indices_word = [all_nodes.index(x) for x in word_nodes]
    indices_entity = [all_nodes.index(x) for x in entity_nodes]
    del tid_nodes
    del userid_nodes
    del word_nodes
    del entity_nodes
    mins = (time() - start) / 60
    print('\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # ----------------------tweet-user-tweet----------------------
    print('\tStart constructing tweet-user-tweet commuting matrix ...')
    print('\t\t\tStart constructing tweet-user matrix ...')
    message += '\tStart constructing tweet-user-tweet commuting matrix ...\n\t\t\tStart constructing tweet-user ' \
               'matrix ...\n '
    start = time()
    w_tid_userid = A[np.ix_(indices_tid, indices_userid)]
    #  return a N(indices_tid)*N(indices_userid) matrix, representing the weight of edges between tid and userid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # convert to scipy sparse matrix
    print('\t\t\tConverting to sparse matrix ...')
    message += '\t\t\tConverting to sparse matrix ...\n'
    start = time()
    s_w_tid_userid = sparse.csr_matrix(w_tid_userid)  # matrix compression
    del w_tid_userid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tTransposing ...')
    message += '\t\t\tTransposing ...\n'
    start = time()
    s_w_userid_tid = s_w_tid_userid.transpose()
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tCalculating tweet-user * user-tweet ...')
    message += '\t\t\tCalculating tweet-user * user-tweet ...\n'
    start = time()
    s_m_tid_userid_tid = s_w_tid_userid * s_w_userid_tid  # homogeneous message graph
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tSaving ...')
    message += '\t\t\tSaving ...\n'
    start = time()
    if save_path is not None:
        sparse.save_npz(save_path + "s_m_tid_userid_tid.npz", s_m_tid_userid_tid)
        print("Sparse binary userid commuting matrix saved.")
        del s_m_tid_userid_tid
    del s_w_tid_userid
    del s_w_userid_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # ----------------------tweet-ent-tweet------------------------
    print('\tStart constructing tweet-ent-tweet commuting matrix ...')
    print('\t\t\tStart constructing tweet-ent matrix ...')
    message += '\tStart constructing tweet-ent-tweet commuting matrix ...\n\t\t\tStart constructing tweet-ent matrix ' \
               '...\n '
    start = time()
    w_tid_entity = A[np.ix_(indices_tid, indices_entity)]
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # convert to scipy sparse matrix
    print('\t\t\tConverting to sparse matrix ...')
    message += '\t\t\tConverting to sparse matrix ...\n'
    start = time()
    s_w_tid_entity = sparse.csr_matrix(w_tid_entity)
    del w_tid_entity
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tTransposing ...')
    message += '\t\t\tTransposing ...\n'
    start = time()
    s_w_entity_tid = s_w_tid_entity.transpose()
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tCalculating tweet-ent * ent-tweet ...')
    message += '\t\t\tCalculating tweet-ent * ent-tweet ...\n'
    start = time()
    s_m_tid_entity_tid = s_w_tid_entity * s_w_entity_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tSaving ...')
    message += '\t\t\tSaving ...\n'
    start = time()
    if save_path is not None:
        sparse.save_npz(save_path + "s_m_tid_entity_tid.npz", s_m_tid_entity_tid)
        print("Sparse binary entity commuting matrix saved.")
        del s_m_tid_entity_tid
    del s_w_tid_entity
    del s_w_entity_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # ----------------------tweet-word-tweet----------------------
    print('\tStart constructing tweet-word-tweet commuting matrix ...')
    print('\t\t\tStart constructing tweet-word matrix ...')
    message += '\tStart constructing tweet-word-tweet commuting matrix ...\n\t\t\tStart constructing tweet-word ' \
               'matrix ...\n '
    start = time()
    w_tid_word = A[np.ix_(indices_tid, indices_word)]
    del A
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # convert to scipy sparse matrix
    print('\t\t\tConverting to sparse matrix ...')
    message += '\t\t\tConverting to sparse matrix ...\n'
    start = time()
    s_w_tid_word = sparse.csr_matrix(w_tid_word)
    del w_tid_word
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tTransposing ...')
    message += '\t\t\tTransposing ...\n'
    start = time()
    s_w_word_tid = s_w_tid_word.transpose()
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tCalculating tweet-word * word-tweet ...')
    message += '\t\t\tCalculating tweet-word * word-tweet ...\n'
    start = time()
    s_m_tid_word_tid = s_w_tid_word * s_w_word_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tSaving ...')
    message += '\t\t\tSaving ...\n'
    start = time()
    if save_path is not None:
        sparse.save_npz(save_path + "s_m_tid_word_tid.npz", s_m_tid_word_tid)
        print("Sparse binary word commuting matrix saved.")
        del s_m_tid_word_tid
    del s_w_tid_word
    del s_w_word_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # ----------------------compute tweet-tweet adjacency matrix----------------------
    print('\tComputing tweet-tweet adjacency matrix ...')
    message += '\tComputing tweet-tweet adjacency matrix ...\n'
    start = time()
    if save_path is not None:
        s_m_tid_userid_tid = sparse.load_npz(save_path + "s_m_tid_userid_tid.npz")
        print("Sparse binary userid commuting matrix loaded.")
        s_m_tid_entity_tid = sparse.load_npz(save_path + "s_m_tid_entity_tid.npz")
        print("Sparse binary entity commuting matrix loaded.")
        s_m_tid_word_tid = sparse.load_npz(save_path + "s_m_tid_word_tid.npz")
        print("Sparse binary word commuting matrix loaded.")

    s_A_tid_tid = s_m_tid_userid_tid + s_m_tid_entity_tid
    del s_m_tid_userid_tid
    del s_m_tid_entity_tid
    s_bool_A_tid_tid = (s_A_tid_tid + s_m_tid_word_tid).astype('bool')  # confirm the connect between tweets
    del s_m_tid_word_tid
    del s_A_tid_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'
    all_mins = (time() - all_start) / 60
    print('\tOver all time elapsed: ', all_mins, ' mins\n')
    message += '\tOver all time elapsed: '
    message += str(all_mins)
    message += ' mins\n'

    if save_path is not None:
        sparse.save_npz(save_path + "s_bool_A_tid_tid.npz", s_bool_A_tid_tid)
        print("Sparse binary adjacency matrix saved.")
        s_bool_A_tid_tid = sparse.load_npz(save_path + "s_bool_A_tid_tid.npz")
        print("Sparse binary adjacency matrix loaded.")

    # create corresponding dgl graph
    G = dgl.DGLGraph(s_bool_A_tid_tid)
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())
    print()
    message += 'We have '
    message += str(G.number_of_nodes())
    message += ' nodes.'
    message += 'We have '
    message += str(G.number_of_edges())
    message += ' edges.\n'

    return all_mins, message

# Split the Twitter dataset by date into 21 message blocks, use the message blocks to construct heterogeneous social graphs,
# and maps them into homogeneous message graphs. 
# Note that:
# 1) We adopt the Latest Message Strategy (which is the most efficient and gives the strongest performance. See Section 4.4 of the paper for more details) here, 
# as a consequence, each message graph only contains the messages of the date and all previous messages are removed from the graph;
# To switch to the All Message Strategy or the Relevant Message Strategy, replace 'G = construct_graph_from_df(incr_df)' with 'G = construct_graph_from_df(incr_df, G)'.
# 2) For test purpose, set test=True, and the message blocks, as well as the resulted message graphs each will contain 100 messages.
# To use all the messages, set test=False, and the number of messages in the message blocks will follow Table. 4 of the paper.
def construct_incremental_dataset_0922(df, save_path, features, test=True):
    # If test equals true, construct the initial graph using test_ini_size tweets
    # and increment the graph by test_incr_size tweets each day
    test_ini_size = 500
    test_incr_size = 100

    # save data splits for training/validate/test mask generation
    # data_split = []
    # save time spent for the heterogeneous -> homogeneous conversion of each graph
    all_graph_mins = []
    message = ""
    # extract distinct dates
    distinct_dates = df.date.unique()  # 2012-11-07
    # print("Distinct dates: ", distinct_dates)
    print("Number of distinct dates: ", len(distinct_dates))
    print()
    message += "Number of distinct dates: "
    message += str(len(distinct_dates))
    message += "\n"

    # split data by dates and construct graphs
    # first week -> initial graph (20254 tweets)
    print("Start constructing initial graph ...")
    message += "\nStart constructing initial graph ...\n"
    # ini_df = df.loc[df['date'].isin(distinct_dates[:7])]  # find top 7 dates
    # if test:
    #     ini_df = ini_df[:test_ini_size]  # top test_ini_size dates
    ini_df = df
    G = construct_graph_from_df(ini_df)
    path = save_path + '0/'
    os.mkdir(path)
    grap_mins, graph_message = to_dgl_graph_v3(G, save_path=path)
    message += graph_message
    print("Initial graph saved")
    message += "Initial graph saved\n"
    # record the total number of tweets
    # data_split.append(ini_df.shape[0])
    # record the time spent for graph conversion
    all_graph_mins.append(grap_mins)
    # extract and save the labels of corresponding tweets
    y = ini_df['event_id'].values
    y = [int(each) for each in y]
    np.save(path + 'labels.npy', np.asarray(y))
    print("Labels saved.")
    message += "Labels saved.\n"
    # extract and save the features of corresponding tweets
    indices = ini_df['index'].values.tolist()
    x = features[indices, :]
    np.save(path + 'features.npy', x)
    print("Features saved.")
    message += "Features saved.\n\n"

    return message, all_graph_mins


#save_path = './incremental_0808/'
save_path = 'offline_1115/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

# load data (68841 tweets, multiclasses filtered)
df_np = np.load('pre_data/68841_tweets_multiclasses_filtered_0722.npy', allow_pickle=True)
print("Data loaded.")
df = pd.DataFrame(data=df_np, columns=["event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",
                                       "place_type", "place_full_name", "place_country_code", "hashtags",
                                       "user_mentions", "image_urls", "entities",
                                       "words", "filtered_words", "sampled_words"])
print("Data converted to dataframe.")
# print(df.shape)
# print(df.head(10))
# print()

# sort data by time
df = df.sort_values(by='created_at').reset_index()

# append date
df['date'] = [d.date() for d in df['created_at']]

# print(df['date'].value_counts())

# load features
# the dimension of feature is 300 in this dataset
f = np.load('pre_data/features_69612_0709_spacy_lg_zero_multiclasses_filtered.npy')

# generate test graphs, features, and labels
message, all_graph_mins = construct_incremental_dataset_0922(df, save_path, f, True)
with open(save_path + "node_edge_statistics.txt", "w") as text_file:
    text_file.write(message)
np.save(save_path + 'all_graph_mins.npy', np.asarray(all_graph_mins))
print("Time sepnt on heterogeneous -> homogeneous graph conversions: ", all_graph_mins)
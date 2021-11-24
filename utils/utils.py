import numpy as np
import torch
from scipy import sparse
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from torch.utils.data import Dataset


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def run_kmeans(extract_features, extract_labels, indices, isoPath=None):
    # Extract the features and labels of the test tweets
    indices = indices.cpu().detach().numpy()

    if isoPath is not None:
        # Remove isolated points
        temp = torch.load(isoPath)
        temp = temp.cpu().detach().numpy()
        non_isolated_index = list(np.where(temp != 1)[0])
        indices = intersection(indices, non_isolated_index)

    # Extract labels
    extract_labels = extract_labels.cpu().numpy()
    labels_true = extract_labels[indices]

    # Extract features
    # X = extract_features[indices, :]
    X = extract_features.cpu().detach().numpy()
    assert labels_true.shape[0] == X.shape[0]
    n_test_tweets = X.shape[0]  # 100

    # Get the total number of classes
    n_classes = len(set(labels_true.tolist()))

    # k-means clustering
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
    labels = kmeans.labels_

    nmi = metrics.normalized_mutual_info_score(labels_true, labels)
    ami = metrics.adjusted_mutual_info_score(labels_true, labels)
    ari = metrics.adjusted_rand_score(labels_true, labels)

    # Return number of test tweets, number of classes covered by the test tweets, and kMeans cluatering NMI
    return n_test_tweets, n_classes, nmi, ami, ari

def evaluate(extract_features, extract_labels, indices,
             epoch, num_isolated_nodes, save_path, is_validation=True, 
             cluster_type='kmeans'):

    message = ''
    message += '\nEpoch '
    message += str(epoch)
    message += '\n'

    # with isolated nodes
    if cluster_type == 'kmeans':
        n_tweets, n_classes, nmi, ami, ari = run_kmeans(extract_features, extract_labels, indices)
    elif cluster_type == 'dbscan':
        pass

    if is_validation:
        mode = 'validation'
    else:
        mode = 'test'
    message += '\tNumber of ' + mode + ' tweets: '
    message += str(n_tweets)
    message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
    message += str(n_classes)
    message += '\n\t' + mode + ' NMI: '
    message += str(nmi)
    message += '\n\t' + mode + ' AMI: '
    message += str(ami)
    message += '\n\t' + mode + ' ARI: '
    message += str(ari)
    if cluster_type == 'dbscan':
        message += '\n\t' + mode + ' best_eps: '
        # message += str(best_eps)
        message += '\n\t' + mode + ' best_min_Pts: '
        # message += str(best_min_Pts)

    if num_isolated_nodes != 0:
        # without isolated nodes
        message += '\n\tWithout isolated nodes:'
        n_tweets, n_classes, nmi, ami, ari = run_kmeans(extract_features, extract_labels, indices,
                                                        save_path + '/isolated_nodes.pt')
        message += '\tNumber of ' + mode + ' tweets: '
        message += str(n_tweets)
        message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
        message += str(n_classes)
        message += '\n\t' + mode + ' NMI: '
        message += str(nmi)
        message += '\n\t' + mode + ' AMI: '
        message += str(ami)
        message += '\n\t' + mode + ' ARI: '
        message += str(ari)
    message += '\n'

    with open(save_path + '/evaluate.txt', 'a') as f:
        f.write(message)
    print(message)

    np.save(save_path + '/%s_metric.npy' % mode, np.asarray([nmi, ami, ari]))

    return nmi


def generateMasks(length, data_split, train_i, i, validation_percent=0.2, save_path=None, remove_obsolete=2):
    """
    Intro:
    This function generates train and validation indices for initial/maintenance epochs and test indices for inference(prediction) epochs
    If remove_obsolete mode 0 or 1:
    For initial/maintenance epochs:
    - The first (train_i + 1) blocks (blocks 0, ..., train_i) are used as training set (with explicit labels)
    - Randomly sample validation_percent of the training indices as validation indices
    For inference(prediction) epochs:
    - The (i + 1)th block (block i) is used as test set
    Note that other blocks (block train_i + 1, ..., i - 1) are also in the graph (without explicit labels, only their features and structural info are leveraged)
    If remove_obsolete mode 2:
    For initial/maintenance epochs:
    - The (i + 1) = (train_i + 1)th block (block train_i = i) is used as training set (with explicit labels)
    - Randomly sample validation_percent of the training indices as validation indices
    For inference(prediction) epochs:
    - The (i + 1)th block (block i) is used as test set

    :param length: the length of label list
    :param data_split: loaded splited data (generated in custom_message_graph.py)
    :param train_i, i: flag, indicating for initial/maintenance stage if train_i == i and inference stage for others
    :param validation_percent: the percent of validation data occupied in whole dataset
    :param save_path: path to save data
    :param num_indices_to_remove: number of indices ought to be removed
    :returns train indices, validation indices or test indices
    """

    # step1: verify total number of nodes
    assert length == data_split[i]  # 500

       # step2.0: if is in initial/maintenance epochs, generate train and validation indices
    if train_i == i:
        # step3: randomly shuffle the graph indices
        train_indices = torch.randperm(length)

        # step4: get total number of validation indices
        n_validation_samples = int(length * validation_percent)

        # step5: sample n_validation_samples validation indices and use the rest as training indices
        validation_indices = train_indices[:n_validation_samples]
        train_indices = train_indices[n_validation_samples:]

        # print(save_path) #./incremental_0808//embeddings_0403100832/block_0/masks
        # step6: save indices
        if save_path is not None:
            torch.save(train_indices, save_path + '/train_indices.pt')
            torch.save(validation_indices, save_path + '/validation_indices.pt')
            
        return train_indices, validation_indices
        # step2.1: if is in inference(prediction) epochs, generate test indices
    else:
        test_indices = torch.arange(0, (data_split[i]), dtype=torch.long)
        if save_path is not None:
            torch.save(test_indices, save_path + '/test_indices.pt')

        return test_indices

def gen_offline_masks(length, validation_percent=0.2, test_percent=0.1):

    test_length = int(length * test_percent)
    valid_length = int(length * validation_percent)
    train_length = length - valid_length - test_length

    samples = torch.randperm(length)
    train_indices = samples[:train_length]
    valid_indices = samples[train_length:train_length + valid_length]
    test_indices = samples[train_length + valid_length:]


    return train_indices, valid_indices, test_indices


def save_embeddings(extracted_features, save_path):
    
    torch.save(extracted_features, save_path + '/final_embeddings.pt')
    print('extracted features saved.')

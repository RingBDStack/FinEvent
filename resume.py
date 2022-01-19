import numpy as np
import json
import argparse
import torch
from time import localtime, strftime
import os

from stage import FinEvent
from utils.metrics import AverageNonzeroTripletsMetric
from utils.utils import *
from layers.TripletLoss import *
from model.MarGNN import MarGNN

def args_register():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', default=50, type=int, help="Number of initial-training/maintenance-training epochs.")
    parser.add_argument('--window_size', default=3, type=int, help="Maintain the model after predicting window_size blocks.")
    parser.add_argument('--patience', default=5, type=int, 
                        help="Early stop if performance did not improve in the last patience epochs.")
    parser.add_argument('--margin', default=3., type=float, help="Margin for computing triplet losses")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate")
    parser.add_argument('--batch_size', default=100, type=int,
                        help="Batch size (number of nodes sampled to compute triplet loss in each batch)")
    parser.add_argument('--hidden_dim', default=128, type=int, help="Hidden dimension")
    parser.add_argument('--out_dim', default=64, type=int, help="Output dimension of tweet representations")
    parser.add_argument('--heads', default=4, type=int, help="Number of heads used in GAT")
    parser.add_argument('--validation_percent', default=0.2, type=float, help="Percentage of validation nodes(tweets)")
    parser.add_argument('--use_hardest_neg', dest='use_hardest_neg', default=False, action='store_true',
                        help="If true, use hardest negative messages to form triplets. Otherwise use random ones")
    parser.add_argument('--is_shared', default=False)
    parser.add_argument('--inter_opt', default='cat_w_avg')
    parser.add_argument('--is_initial', default=False)
    parser.add_argument('--sampler', default='RL_sampler')
    parser.add_argument('--cluster_type', default='kmeans', help="Types of clustering algorithms")  # dbscan

    # RL-0
    parser.add_argument('--threshold_start0', default=[[0.5],[0.5],[0.5]], type=float,
                        help="The initial value of the filter threshold for state1 or state3")
    parser.add_argument('--RL_step0', default=0.02, type=float,
                        help="The step size of RL for state1 or state3")
    parser.add_argument('--RL_start0', default=0, type=int,
                        help="The starting epoch of RL for state1 or state3")

    # RL-1
    parser.add_argument('--eps_start', default=0.001, type=float,
                        help="The initial value of the eps for state2")
    parser.add_argument('--eps_step', default=0.02, type=float,
                        help="The step size of eps for state2")
    parser.add_argument('--min_Pts_start', default=2, type=int,
                        help="The initial value of the min_Pts for state2")
    parser.add_argument('--min_Pts_step', default=1, type=int,
                        help="The step size of min_Pts for state2")

    # other arguments
    parser.add_argument('--use_cuda', dest='use_cuda', default=True, 
                        action='store_true', help="Use cuda")
    parser.add_argument('--data_path', default='./incremental_0502/', type=str,
                        help="Path of features, labels and edges")
    # format: './incremental_0808/incremental_graphs_0808/embeddings_XXXX'
    parser.add_argument('--mask_path', default=None, type=str,
                        help="File path that contains the training, validation and test masks")
    # format: './incremental_0808/incremental_graphs_0808/embeddings_XXXX'
    parser.add_argument('--resume_path', default='incremental_cross_English_68841/', type=str,
                        help='Resume trained model and directly used to inference')
    parser.add_argument('--log_interval', default=10, type=int,
                        help="Log interval")
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    # define args
    args = args_register()

    # check CUDA
    print('Using CUDA:', torch.cuda.is_available())

    # create working path
    embedding_save_path = args.data_path + '/embeddings_' + strftime("%m%d%H%M%S", localtime())
    os.mkdir(embedding_save_path)
    print('embedding save path: ', embedding_save_path)

    print('Batch Size:', args.batch_size)
    print('Intra Agg Mode:', args.is_shared)
    print('Inter Agg Mode:', args.inter_opt)
    print('Reserve node config?', args.is_initial)

    print('Trained model from %s dataset' % args.resume_path)
    print('Inference dataset:', args.data_path)

    # record hyper-parameters
    with open(embedding_save_path + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    # load number of messages in each blocks
    # e.g. data_split = [  500  ,   100, ...,  100]
    #                    block_0  block_1    block_n
    data_split = np.load(args.data_path + '/data_split.npy')

    # define loss function
    # contrastive loss in our paper
    if args.use_hardest_neg:
        loss_fn = OnlineTripletLoss(args.margin, HardestNegativeTripletSelector(args.margin))
    else:
        loss_fn = OnlineTripletLoss(args.margin, RandomNegativeTripletSelector(args.margin))

    # define metrics
    BCL_metrics = [AverageNonzeroTripletsMetric()]

    # define detection stage
    Streaming = FinEvent(args)

    # pre-train stage: train on initial graph
    train_i = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MarGNN((302, args.hidden_dim, args.out_dim, args.heads), 
            num_relations=3, inter_opt=args.inter_opt, is_shared=args.is_shared)
    model.load_state_dict(torch.load(args.resume_path + '/block_18/models/best.pt'))

    model, RL_thresholds = Streaming.initial_maintain(train_i=train_i,
                                                     i=0,
                                                     metrics=BCL_metrics,
                                                     embedding_save_path=embedding_save_path,
                                                     loss_fn=loss_fn,
                                                     model=model)

    # detection-maintenance stage: incremental training and detection
    for i in range(1, data_split.shape[0]):
        # infer every block
        model = Streaming.inference(train_i=train_i,
                                    i=i,
                                    metrics=BCL_metrics,
                                    embedding_save_path=embedding_save_path,
                                    loss_fn=loss_fn,
                                    model=model,
                                    RL_thresholds=RL_thresholds)
    
        # maintenance in window size and desert the last block
        # if i % args.window_size == 0 and i != data_split.shape[0] - 1:
        #     train_i = i
        #     model.load_state_dict(torch.load(args.resume_path + '/block_%d/models/best.pt' % i))
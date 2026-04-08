import torch
import time
import sys
import os
import math
import argparse
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data as Data

np_load_old = np.load
np.aload = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
from gnia import GNIA

sys.path.append('..')
from utils import *
from model import oriRGCN, num_decoder, cat_decoder, SHGNDetector, encoder, num_encoder, RGCN_weight
from Dataset import Twibot22
import torch
from torch import nn
from utils import accuracy, init_weights

from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc

from torch_geometric.loader import NeighborLoader
# from torch_geometric.data import Data, HeteroData

import pandas as pd
import numpy as np


def main(opts):
    # hyperparameters
    gpu_id = opts['gpu']
    seed = opts['seed']
    connect = opts['connect']
    multi = opts['multiedge']
    discrete = opts['discrete']
    suffix = opts['suffix']
    attr_tau = float(opts['attrtau']) if opts['attrtau'] != None else opts['attrtau']
    edge_tau = float(opts['edgetau']) if opts['edgetau'] != None else opts['edgetau']
    lr = opts['lr']
    patience = opts['patience']
    best_score = opts['best_score']
    counter = opts['counter']
    nepochs = opts['nepochs']
    st_epoch = opts['st_epoch']
    epsilon_start = opts['epsst']
    epsilon_end = 0
    epsilon_decay = opts['epsdec']
    total_steps = 500
    batch_size = opts['batchsize']

    ckpt_save_dirs = 'checkpoint/bot_gnia/'
    model_save_file = ckpt_save_dirs
    if not os.path.exists(ckpt_save_dirs):
        os.makedirs(ckpt_save_dirs)

    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # # 修改后
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        torch.cuda.set_device(0)  # 显式设置使用第一个可见设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # root = './processed_data/'
    # dataset = Twibot22(root=root, device=device, process=False, save=False)
    # des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels_np, train_mask, val_mask, test_mask = dataset.dataloader()

    des_tensor = torch.load('./processed_data/des_tensor.pt')
    tweets_tensor = torch.load('./processed_data/tweets_tensor.pt')
    num_prop = torch.load('./processed_data/num_properties_tensor.pt')
    category_prop = torch.load('./processed_data/cat_properties_tensor.pt')
    edge_index = torch.load('./processed_data/edge_index.pt')
    edge_type = torch.load('./processed_data/edge_type.pt')
    labels_np = torch.load('./processed_data/label.pt')
    train_mask = torch.load('./processed_data/train_idx.pt')
    val_mask = torch.load('./processed_data/val_idx.pt')
    test_mask = torch.load('./processed_data/test_idx.pt')

    des_tensor = des_tensor.to(device)
    tweets_tensor = tweets_tensor.to(device)
    num_prop = num_prop.to(device)
    category_prop = category_prop.to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    labels_np = labels_np.to(device)
    train_mask = train_mask
    val_mask = val_mask
    test_mask = test_mask

    train_mask_human = []
    train_mask_bot = []
    for i in train_mask:
        if labels_np[i] == 0:
            train_mask_human.append(i)
        else:
            train_mask_bot.append(i)

    val_mask_human = []
    val_mask_bot = []
    for i in val_mask:
        if labels_np[i] == 0:
            val_mask_human.append(i)
        else:
            val_mask_bot.append(i)

    test_mask_human = []
    test_mask_bot = []
    for i in test_mask:
        if labels_np[i] == 0:
            test_mask_human.append(i)
        else:
            test_mask_bot.append(i)

    human_mask = np.append(train_mask_human, val_mask_human, axis=0)
    human_mask = np.append(human_mask, test_mask_human, axis=0)
    bot_mask = np.append(train_mask_bot, val_mask_bot, axis=0)
    bot_mask = np.append(bot_mask, test_mask_bot, axis=0)

    train_mask_human = torch.tensor(train_mask_human)
    train_mask_bot = torch.tensor(train_mask_bot)
    val_mask_human = torch.tensor(val_mask_human)
    val_mask_bot = torch.tensor(val_mask_bot)
    test_mask_human = torch.tensor(test_mask_human)
    test_mask_bot = torch.tensor(test_mask_bot)
    human_mask = torch.tensor(human_mask)
    bot_mask = torch.tensor(bot_mask)

    ori_model = oriRGCN().to(device)
    ori_model.load_state_dict(torch.load('./processed_data/model_ori_rgcn.pth', map_location='cpu'))

    encoder_model = encoder().to(device)

    model_dict = ori_model.state_dict()
    des_weight = model_dict['linear_relu_des.0.weight'].clone().detach()
    des_bias = model_dict['linear_relu_des.0.bias'].clone().detach()
    tweet_weight = model_dict['linear_relu_tweet.0.weight'].clone().detach()
    tweet_bias = model_dict['linear_relu_tweet.0.bias'].clone().detach()
    num_weight = model_dict['linear_relu_num_prop.0.weight'].clone().detach()
    num_bias = model_dict['linear_relu_num_prop.0.bias'].clone().detach()
    cat_weight = model_dict['linear_relu_cat_prop.0.weight'].clone().detach()
    cat_bias = model_dict['linear_relu_cat_prop.0.bias'].clone().detach()
    encoder_dict = {'linear_relu_des.0.weight': des_weight, 'linear_relu_des.0.bias': des_bias,
                    'linear_relu_tweet.0.weight': tweet_weight, 'linear_relu_tweet.0.bias': tweet_bias,
                    'linear_relu_num_prop.0.weight': num_weight, 'linear_relu_num_prop.0.bias': num_bias,
                    'linear_relu_cat_prop.0.weight': cat_weight, 'linear_relu_cat_prop.0.bias': cat_bias
                    }
    encoder_model.load_state_dict(encoder_dict)

    num_decoder_model = num_decoder().to(device)
    num_decoder_model.load_state_dict(torch.load('./processed_data/model_num_decoder.pth', map_location='cpu'))
    cat_decoder_model = cat_decoder().to(device)
    cat_decoder_model.load_state_dict(torch.load('./processed_data/model_cat_decoder.pth', map_location='cpu'))

    features = encoder_model(des_tensor, tweets_tensor, num_prop, category_prop)
    features = torch.tensor(features, device='cpu')
    features = np.array(features)
    features = sp.csr_matrix(features, dtype=np.float32)

    num_dict = np.load('./processed_data/num_properties_meanstd.npy', allow_pickle=True).item()
    num_mean = torch.tensor(
        [num_dict['followers_count_mean'], num_dict['active_days_mean'], num_dict['screen_name_length_mean'],
         num_dict['following_count_mean'], num_dict['statues_mean']], device=device)
    num_std = torch.tensor(
        [num_dict['followers_count_std'], num_dict['active_days_std'], num_dict['screen_name_length_std'],
         num_dict['following_count_std'], num_dict['statues_std']], device=device)


    adj = load_adj(edge_index)
    n = adj.shape[0]
    nc = labels_np.max() + 1
    nfeat = features.shape[1]
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(n)
    adj[adj > 1] = 1

    if connect:
        lcc = largest_connected_components(adj)
        adj = adj[lcc][:, lcc]
        features = features[lcc]
        labels_np = labels_np[lcc]
        n = adj.shape[0]
        print('Nodes num:', n)

    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    nor_adj_tensor = normalize_tensor(adj_tensor)

    feat = torch.from_numpy(features.todense().astype('double')).float().to(device)
    # second_largest, _ = torch.topk(feat, k=2, dim=0, largest=False)
    feat_max = feat.max(0).values
    feat_min = feat.min(0).values
    labels = labels_np
    degree = adj.sum(1)
    deg = torch.FloatTensor(degree).flatten().to(device)
    feat_num = int(features.sum(1).mean())
    eps_threshold = [epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * steps / epsilon_decay) for steps in
                     range(total_steps)]

    detect_model = SHGNDetector().to(device)
    detect_model.load_state_dict(torch.load('./processed_data/model_SimpleHGN.pth', map_location='cpu'))


    detect_model.eval()
    for p in detect_model.parameters():
        p.requires_grad = False

    node_emb = detect_model(des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type)

    RGCN_weight_model = RGCN_weight().to(device)
    RGCN_weight_model.load_state_dict(torch.load('./processed_data/model_RGCN_weight.pth', map_location='cpu'))

    W_weight = ori_model.rgcn.weight.data.detach()
    W1 = W_weight[0]
    W2 = W_weight[1]
    W = torch.cat((W1, W2), dim=0).t()
    W = RGCN_weight_model(W)


    logits = detect_model(des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type)
    logp = F.log_softmax(logits, dim=1)
    acc = accuracy(logp, labels)
    print('Acc:', acc)
    print('Train Acc:', accuracy(logp[train_mask], labels[train_mask]))
    print('Valid Acc:', accuracy(logp[val_mask], labels[val_mask]))
    print('Test Acc:', accuracy(logp[test_mask], labels[test_mask]))
    print('*' * 30)

    # Initialization
    model = GNIA(labels, nfeat, W, discrete, device, feat_min=feat_min, feat_max=feat_max, feat_num=feat_num,
                 attr_tau=attr_tau, edge_tau=edge_tau).to(device)

    # Test Part
    names = locals()
    training = False
    model.load_state_dict(torch.load(model_save_file + 'checkpoint.pt', map_location='cpu'))
    for p in model.parameters():
        p.requires_grad = False
    for dset in ['test']:
        names[dset + '_bot_atk'] = []
        names[dset + '_new_node'] = []
        for batch in names[dset + '_mask_bot']:
            target = np.array([batch])
            target_deg = int(sum([degree[i].item() for i in target]))
            budget = int(min(round(target_deg / 2), round(degree.mean()))) if multi else 1
            ori = labels_np[target].item()
            best_wrong_label = 0
            one_order_nei = adj[target].nonzero()[1]
            tar_norm_adj = nor_adj_tensor[target.item()].to_dense()
            norm_a_target = tar_norm_adj[one_order_nei].unsqueeze(1)
            inj_feat, disc_score, masked_score_idx = model(target, one_order_nei, budget, feat, norm_a_target, node_emb,
                                                           W[ori], W[best_wrong_label], train_flag=training,
                                                           eps=epsilon_end)
            new_num = num_decoder_model(inj_feat[16:24])
            new_number = (new_num * num_std + num_mean).abs().round()
            new_number[0] = torch.as_tensor(0).reshape(1, 1)
            new_number[1] = torch.as_tensor(100 if new_number[1] > 100 else new_number[1]).reshape(1, 1)
            new_number[2] = torch.as_tensor(15 if new_number[2] > 15 else (1 if new_number[2] < 1 else new_number[2])).reshape(1, 1)
            new_number[3] = torch.as_tensor(
                5000 if new_number[3] > 5000 else new_number[3]).reshape(1, 1)
            new_number[4] = torch.as_tensor(
                500 if new_number[4] > 500 else new_number[4]).reshape(1, 1)
            new_num = (new_number - num_mean) / num_std
            new_cat_float = cat_decoder_model(inj_feat[24:])
            new_cat = torch.tensor(0 if new_cat_float <= 0.5 else 1, device=device).reshape(1, 1)

            new_des_tensor = torch.cat((des_tensor, torch.zeros(1, 768).to(device)), 0)
            new_tweets_tensor = torch.cat((tweets_tensor, torch.zeros(1, 768).to(device)), 0)
            new_num_prop = torch.cat((num_prop, new_num.unsqueeze(0)), 0)
            new_category_prop = torch.cat((category_prop, new_cat), 0)

            new_edge_index = gen_extend_edge_index(edge_index, adj_tensor.shape[0], disc_score, masked_score_idx, device)
            extend_edge_type = torch.tensor([1]).to(device)
            new_edge_type = torch.cat((edge_type, extend_edge_type), 0)

            new_logits = detect_model(new_des_tensor, new_tweets_tensor, new_num_prop, new_category_prop,
                                      new_edge_index, new_edge_type)

            if 0 == new_logits[target].argmax(1).item():
                names[dset + '_bot_atk'].append(1)
            else:
                names[dset + '_bot_atk'].append(0)

            new_node_tar = np.array([-1])

            if 1 == new_logits[new_node_tar].argmax(1).item():
                names[dset + '_new_node'].append(1)
            else:
                names[dset + '_new_node'].append(0)

            del new_logits

        print('Hidden Bot Ratio of ' + dset + ' set:', np.array(names[dset + '_bot_atk']).mean())
        print('New Node Become Bot Ratio of ' + dset + ' set:', np.array(names[dset + '_new_node']).mean())
        print('*' * 30)


if __name__ == '__main__':
    setup_seed(904)
    parser = argparse.ArgumentParser(description='GNIA')

    # configure
    parser.add_argument('--seed', type=int, default=904, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='GPU ID')
    parser.add_argument('--suffix', type=str, default='_', help='suffix of the checkpoint')

    # dataset
    parser.add_argument('--connect', default=False, type=bool, help='largest connected component')
    parser.add_argument('--multiedge', default=False, type=bool,
                        help='budget of malicious edges connected to injected node')

    # optimization
    parser.add_argument('--optimizer', choices=['Adam', 'RMSprop'], default='RMSprop', help='optimizer')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--wd', default=0., type=float, help='weight decay')
    parser.add_argument('--nepochs', type=int, default=10000, help='number of epochs')
    parser.add_argument('--patience', default=20, type=int, help='patience of early stopping')
    parser.add_argument('--batchsize', type=int, default=16, help='batchsize')

    # Hyperparameters
    parser.add_argument('--attrtau', default=None,
                        help='tau of gumbel softmax on attribute on discrete attributed graph')
    parser.add_argument('--edgetau', default=0.01, help='tau of gumbel softmax on edge')
    parser.add_argument('--epsdec', default=1, type=float, help='epsilon decay: coefficient of the gumbel sampling')
    parser.add_argument('--epsst', default=50, type=int, help='epsilon start: coefficient of the gumbel sampling')

    # Ignorable
    parser.add_argument('--counter', type=int, default=0, help='counter for recover training (Ignorable)')
    parser.add_argument('--best_score', type=float, default=0., help='best score for recover training (Ignorable)')
    parser.add_argument('--st_epoch', type=int, default=0, help='start epoch for recover training (Ignorable)')
    parser.add_argument('--local_rank', type=int, default=2, help='DDP local rank for parallel (Ignorable)')

    args = parser.parse_args()
    opts = args.__dict__.copy()
    # opts['discrete'] = False if 'k_' in opts['dataset'] else True
    opts['discrete'] = False
    print(opts)
    att_sucess = main(opts)


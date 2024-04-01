import argparse

import math

import util_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
import random
import losses
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids', type=str, default="1",
                             help='gpu ids to use, -1 for cpu, "0,1" for multi gpu')
args = parser.parse_args()

train_num =7060
label_freq = pd.read_csv('onehot_reuter_train.csv')
#label_freq = label_freq.drop('comment_text', axis=1).drop('id', axis=1).drop('label', axis=1)
class_freq = np.sum(label_freq, axis=0).T
class_freq.tolist()
print(class_freq)

criterion_cb =  util_loss. ResampleLoss(reweight_func=None, loss_weight=1.0,
                             focal=dict(focal=True, alpha=0.5, gamma=2),
                             logit_reg=dict(),
                             class_freq=class_freq, train_num=train_num, reduction='mean')

criterion_cb_nm = criterion_cb =  util_loss. ResampleLoss(reweight_func='CB', loss_weight=10.0,
                             focal=dict(focal=True, alpha=0.5, gamma=2),
                             logit_reg=dict(int_bias=0.05,neg_scale=2.0),
                             CB_loss= dict(CB_beta=0.9,CB_mode='by_class'),
                             class_freq=class_freq, train_num=train_num, reduction='none')
device = torch.device("cpu" if args.gpu_ids[0] == '-1' else "cuda:" + args.gpu_ids[0])
criterion_bce = nn.BCEWithLogitsLoss(size_average=False, reduce=False)
criterion_bce_nm = nn.BCEWithLogitsLoss()


def loss_an1(logits, observed_labels, args):

    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_matrix = criterion_bce(logits, observed_labels)
    corrected_loss_matrix = criterion_bce(logits, torch.logical_not(observed_labels).float())

    return loss_matrix, corrected_loss_matrix

def loss_an2(logits, observed_labels, args):

    assert torch.min(observed_labels) >= 0
    loss_matrix = criterion_cb_nm(logits, observed_labels)
    corrected_loss_matrix = criterion_cb_nm(logits, torch.logical_not(observed_labels).float())

    return loss_matrix, corrected_loss_matrix

def loss_coteaching(y_1, y_2, t, args, epoch):
    k = math.log(((0.9 - (epoch * args.interval)) + 1), 2.5)

    batch_size = int(y_1.size(0))

    num_classes = int(y_1.size(1))


    loss_1, corrected_loss_matrix1 = loss_an1(y_1, t.clamp(0), args)
    loss_2, corrected_loss_matrix2 = loss_an1(y_2, t.clamp(0), args)

    from itertools import chain
    data1 = list(chain.from_iterable(loss_1.cpu().detach().numpy()))
    data1 = np.array(data1).reshape(-1, 1)

    data2 = list(chain.from_iterable(loss_2.cpu().detach().numpy()))
    data2 = np.array(data2).reshape(-1, 1)
    from sklearn.cluster import DBSCAN
    min_dbs = math.ceil(batch_size * num_classes * 0.5)

    topk_high1 = torch.topk(loss_1.flatten(), 2)
    topk_low1 = torch.topk(loss_1.flatten(), 2, largest=False)
    topk1 = (topk_high1.values[-1].cpu().detach().numpy() - topk_low1.values[-1].cpu().detach().numpy())

    dbs1 = DBSCAN(eps=topk1 * k, min_samples=min_dbs)
    dbs1.fit(data1)
    labels1 = dbs1.fit_predict(data1)

    topk_high2 = torch.topk(loss_2.flatten(), 2)
    topk_low2 = torch.topk(loss_2.flatten(), 2, largest=False)
    topk2 = (topk_high2.values[-1].cpu().detach().numpy() - topk_low2.values[-1].cpu().detach().numpy())
    dbs2 = DBSCAN(eps=topk2 * k, min_samples=min_dbs)
    dbs2.fit(data2)
    labels2 = dbs2.fit_predict(data2)

    zero_matrix = torch.zeros_like(loss_1)

    labels1 = torch.tensor(labels1, dtype=torch.float32).reshape(batch_size, num_classes).cuda()
    correction_idx1 = torch.where(labels1 != 0)


    labels2 = torch.tensor(labels2, dtype=torch.float32).reshape(batch_size, num_classes).cuda()
    correction_idx2 = torch.where(labels2 != 0)

    '''
    LC:
    loss_1_update = torch.where(labels2 == 0, loss_1, corrected_loss_matrix1)
    # loss_2_update = torch.where(labels1 == 0, loss_2, corrected_loss_matrix2)
    '''

    # LR:
    loss_1_update = torch.where(labels2 == 0, loss_1, zero_matrix)
    loss_2_update = torch.where(labels1 == 0, loss_2, zero_matrix)

    loss_1_update = loss_1_update.mean()
    loss_2_update = loss_2_update.mean()

    return loss_1_update, loss_2_update
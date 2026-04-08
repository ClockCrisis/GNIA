# coding: utf-8
from model import BotGCN, oriRGCN, HGTDetector, SHGNDetector, RGCN,BotRGCN
from Dataset import Twibot22
import torch
from torch import nn
from utils import bot_accuracy,init_weights

from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve,auc

from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data, HeteroData

import pandas as pd
import numpy as np

device = 'cuda:0'
embedding_size,dropout,lr,weight_decay=32,0.1,1e-3,5e-2

# root='./processed_data/'
# dataset=Twibot22(root=root,device=device,process=False,save=False)

des_tensor = torch.load('./processed_data/des_tensor.pt')
tweets_tensor = torch.load('./processed_data/tweets_tensor.pt')
num_prop = torch.load('./processed_data/num_properties_tensor.pt')
category_prop = torch.load('./processed_data/cat_properties_tensor.pt')
edge_index = torch.load('./processed_data/edge_index.pt')
edge_type = torch.load('./processed_data/edge_type.pt')
labels = torch.load('./processed_data/label.pt')
train_idx = torch.load('./processed_data/train_idx.pt')
val_idx = torch.load('./processed_data/val_idx.pt')
test_idx = torch.load('./processed_data/test_idx.pt')

des_tensor = des_tensor.to(device)
tweets_tensor = tweets_tensor.to(device)
num_prop = num_prop.to(device)
category_prop = category_prop.to(device)
edge_index = edge_index.to(device)
edge_type = edge_type.to(device)
labels = labels.to(device)
train_idx = train_idx.to(device)
val_idx = val_idx.to(device)
test_idx = test_idx.to(device)

# print(num_prop)
# print(qqq)

#修改了模型，但是好像这个RGCN不能直接加载上去啊。
#不过也是，源码只导入了四个模型，不记得我之前有没有跑过这个了。说不定这个model_RGCN.pth是在别处产生的吧,也可能是oriGCN命名错了？
#反正先把剩下这三个模型训练跑一下吧。
#又导入了BotRGCN，说不定是这个？
#model=oriRGCN().to(device)
# model=RGCN().to(device)没有用
# model=BotRGCN().to(device)
model=BotGCN().to(device)
# model=HGTDetector().to(device)
# model=SHGNDetector().to(device)


loss=nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),
                    lr=lr,weight_decay=weight_decay)

def train(epoch):
    model.train()
    output = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index, edge_type)
    loss_train = loss(output[train_idx], labels[train_idx])
    acc_train = bot_accuracy(output[train_idx], labels[train_idx])
    acc_val = bot_accuracy(output[val_idx], labels[val_idx])
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'acc_train: {:.4f}'.format(acc_train.item()),
        'acc_val: {:.4f}'.format(acc_val.item()),)
    return acc_train,loss_train

# def test():
#     model.eval()
#     output = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index, edge_type)
#     loss_test = loss(output[test_idx], labels[test_idx])
#     acc_test = bot_accuracy(output[test_idx], labels[test_idx])
#     output=output.max(1)[1].to('cpu').detach().numpy()
#     label=labels.to('cpu').detach().numpy()
#     f1=f1_score(label[test_idx],output[test_idx])
#     precision=precision_score(label[test_idx],output[test_idx])
#     recall=recall_score(label[test_idx],output[test_idx])
#     fpr, tpr, thresholds = roc_curve(label[test_idx], output[test_idx], pos_label=1)
#     Auc=auc(fpr, tpr)
#     print("test set results:",
#             "test_loss= {:.4f}".format(loss_test.item()),
#             "test_accuracy= {:.4f}".format(acc_test.item()),
#             "precision= {:.4f}".format(precision.item()),
#             "recall= {:.4f}".format(recall.item()),
#             "f1_score= {:.4f}".format(f1.item()),
#             "auc= {:.4f}".format(Auc.item()),
#             )

#test也做了重写
def test():
    model.eval()
    with torch.no_grad():
        output = model(des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type)
        loss_test = loss(output[test_idx], labels[test_idx])
        acc_test = bot_accuracy(output[test_idx], labels[test_idx])

        # 将数据移动到CPU并转换为numpy数组
        output_pred = output.argmax(dim=1).cpu().numpy()
        output_prob = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
        label_cpu = labels.cpu().numpy()
        test_idx_cpu = test_idx.cpu().numpy()

        # 计算指标
        f1 = f1_score(label_cpu[test_idx_cpu], output_pred[test_idx_cpu])
        precision = precision_score(label_cpu[test_idx_cpu], output_pred[test_idx_cpu])
        recall = recall_score(label_cpu[test_idx_cpu], output_pred[test_idx_cpu])
        fpr, tpr, _ = roc_curve(label_cpu[test_idx_cpu], output_prob[test_idx_cpu])
        auc_score = auc(fpr, tpr)

        print("test set results:",
              "test_loss= {:.4f}".format(loss_test.item()),
              "test_accuracy= {:.4f}".format(acc_test.item()),
              "precision= {:.4f}".format(precision),
              "recall= {:.4f}".format(recall),
              "f1_score= {:.4f}".format(f1),
              "auc= {:.4f}".format(auc_score))


model.apply(init_weights)

epochs =150
for epoch in range(epochs):
    train(epoch)


#torch.save(model.state_dict(), './processed_data/model_ori_rgcn.pth')
# torch.save(model.state_dict(), './processed_data/model_RGCN.pth')
torch.save(model.state_dict(), './processed_data/model_GCN.pth')
# torch.save(model.state_dict(), './processed_data/model_SimpleHGN.pth')
# torch.save(model.state_dict(), './processed_data/model_HGT.pth')
test()

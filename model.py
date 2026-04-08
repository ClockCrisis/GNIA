import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import RGCNConv, GCNConv, HGTConv
import torch.nn.functional as F
from layer import SimpleHGN

import numpy as np

class oriRGCN(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=1, embedding_dimension=32,
                 dropout=0.3):
        super(oriRGCN, self).__init__()
        self.dropout = dropout
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )

        self.rgcn = RGCNConv(embedding_dimension, 2, num_relations=2, bias=False)


    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)

        x = self.rgcn(x, edge_index, edge_type)

        return x

class RGCN(nn.Module):
    def __init__(self, embedding_dimension=32, dropout=0.3):
        super(RGCN, self).__init__()
        self.dropout = dropout

        self.rgcn = RGCNConv(embedding_dimension, 2, num_relations=2, bias=False)


    def forward(self, x, edge_index, edge_type):
        x = self.rgcn(x, edge_index, edge_type)

        return x


class BotRGCN(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=1, embedding_dimension=32,
                 dropout=0.3):
        super(BotRGCN, self).__init__()
        self.dropout = dropout
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )

        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)
        # self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)#测试两个RGCN层
        self.gcn1 = GCNConv(embedding_dimension, embedding_dimension)#测试GCN层

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)


    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)

        x = self.linear_relu_input(x)
        #本来是两层
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)

        #测试一层
        # x = self.rgcn(x, edge_index, edge_type)

        #测试GCN层
        # x = self.gcn1(x, edge_index)

        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x


class num_decoder(nn.Module):
    def __init__(self):
        super(num_decoder, self).__init__()
        self.linear_output_1 = nn.Sequential(
            nn.Linear(8, 8),
            nn.LeakyReLU()
        )

        self.linear_output_2 = nn.Sequential(
            nn.Linear(8, 8),
            nn.LeakyReLU()
        )

        self.linear_output_3 = nn.Sequential(
            nn.Linear(8, 5)
        )
    def forward(self, x):
        out = self.linear_output_1(x)
        out = self.linear_output_2(out)
        out = self.linear_output_3(out)
        return out

class cat_decoder(nn.Module):
    def __init__(self):
        super(cat_decoder, self).__init__()
        self.linear_output_1 = nn.Sequential(
            nn.Linear(8, 8),
            nn.LeakyReLU()
        )

        self.linear_output_2 = nn.Sequential(
            nn.Linear(8, 8),
            nn.LeakyReLU()
        )

        self.linear_output_3 = nn.Sequential(
            nn.Linear(8, 1)
        )
    def forward(self, x):
        out = self.linear_output_1(x)
        out = self.linear_output_2(out)
        out = self.linear_output_3(out)
        return out


#这一个GAT_cat——decoder是新加的，用于测试修改后的属性恢复模型
from torch_geometric.nn import GATConv
import torch.nn.functional as F
class GAT_cat_decoder(nn.Module):
    def __init__(self, in_channels=8, hidden_channels=8, out_channels=1, heads=1):
        super(GAT_cat_decoder, self).__init__()
        # 第一层GAT卷积
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.1)
        # 第二层GAT卷积
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=0.1)
        # 输出层（图注意力不适合直接输出scalar，这里使用Linear）
        self.out_linear = nn.Linear(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        # 第一层图注意力 + 激活
        x = self.gat1(x, edge_index)
        x = F.leaky_relu(x)
        # 第二层图注意力 + 激活
        x = self.gat2(x, edge_index)
        x = F.leaky_relu(x)
        # 输出层
        x = self.out_linear(x)
        return x
class GAT_num_decoder1(nn.Module):
    def __init__(self, hidden_dim=128, des_size=768, tweet_size=768,
                 num_prop_size=5, cat_prop_size=1, dropout=0.3, heads=4):
        super(GAT_num_decoder, self).__init__()

        # 每种特征独立线性降维 + 激活
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, hidden_dim // 4),
            nn.LeakyReLU()
        )

        # 图注意力层
        self.gat = GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout, concat=False)

        # 输出映射：结合节点特征 + x 数组（x ∈ ℝ^8），输出 5 维向量
        self.linear_out = nn.Sequential(
            nn.Linear(hidden_dim + 8, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5)  # 输出 5 维
        )

    def forward(self, x, des, tweet, num_prop, cat_prop, edge_index, target_node_idx):
        """
        参数说明：
        - x: Tensor, shape = [8]，特定节点的数值属性向量
        - des, tweet, num_prop, cat_prop: 所有节点的特征 (N, D)
        - edge_index: 边信息 (2, E)
        - target_node_idx: 指定节点索引（用于取出对应节点的表示）
        """

        # 每类特征线性映射
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)

        # 拼接成节点表示
        feature = torch.cat((d, t, n, c), dim=1)  # shape: [N, hidden_dim]
        # 图注意力层
        gat_out = self.gat(feature, edge_index)  # shape: [N, hidden_dim]
        # 取出指定节点表示
        target_feat = gat_out[target_node_idx]  # shape: [hidden_dim]
        # 拼接目标节点表示与其数字属性 x
        combined = torch.cat([target_feat, x], dim=0)  # shape: [hidden_dim + 8]
        # 输出为 5 维向量
        out = self.linear_out(combined)  # shape: [5]
        return out


class GAT_num_decoder(nn.Module):
    def __init__(self, input_dim=8, output_dim=5):
        super(GAT_num_decoder, self).__init__()
        # 这里不需要计算跨样本注意力，因为输入是二维张量，每行独立处理
        self.attn = nn.Linear(input_dim, input_dim)  # 给每个输入特征计算注意力权重向量
        self.fc = nn.Linear(input_dim, output_dim)  # 映射到目标输出维度5

    def forward(self, inputs):
        # 计算注意力分数，注意力是对8维特征做加权
        scores = self.attn(inputs)  # (N, 8)
        # 对每行特征权重做softmax归一化，突出重要特征
        attn_weights = F.softmax(scores, dim=1)  # (N, 8)
        # 将输入乘以注意力权重，逐元素乘法
        weighted_inputs = inputs * attn_weights  # (N, 8)
        # 经过线性层映射成输出维度
        output = self.fc(weighted_inputs)  # (N, 5)
        return output
class encoder(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=1, embedding_dimension=32):
        super(encoder, self).__init__()
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )

    def forward(self, des, tweet, num_prop, cat_prop):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)
        return x


class num_encoder(nn.Module):
    def __init__(self, num_prop_size=5, embedding_dimension=32):
        super(num_encoder, self).__init__()
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
    def forward(self, num_prop):
        n = self.linear_relu_num_prop(num_prop)
        return n


class cat_encoder(nn.Module):
    def __init__(self, cat_prop_size=1, embedding_dimension=32):
        super(cat_encoder, self).__init__()
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
    def forward(self, cat_prop):
        c = self.linear_relu_cat_prop(cat_prop)
        return c


class RGCN_weight(nn.Module):
    def __init__(self):
        super(RGCN_weight, self).__init__()

        self.linear_output = nn.Linear(64, 32)

    def forward(self, x):
        x = self.linear_output(x)

        return x


class BotGCN(nn.Module):
    def __init__(self, hidden_dim=128, des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=1, dropout=0.3):
        super(BotGCN, self).__init__()
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, hidden_dim // 4),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dim, 2)

        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        # self.gcn3 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type=None):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)
        x = self.dropout(x)
        x = self.linear_relu_input(x)
        x = self.gcn1(x, edge_index)
        x = self.dropout(x)
        x = self.gcn2(x, edge_index)
        # x = self.dropout(x)
        # x = self.gcn3(x,edge_index)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x


class HGTDetector(nn.Module):
    def __init__(self, numeric_num=5, cat_num=1, linear_channels=128, tweet_channel=768, des_channel=768, out_channel=128,
                 dropout=0.5):
        super(HGTDetector, self).__init__()

        self.in_linear_numeric = nn.Linear(numeric_num, int(linear_channels / 4), bias=True)
        self.in_linear_bool = nn.Linear(cat_num, int(linear_channels / 4), bias=True)
        self.in_linear_tweet = nn.Linear(tweet_channel, int(linear_channels / 4), bias=True)
        self.in_linear_des = nn.Linear(des_channel, int(linear_channels / 4), bias=True)
        self.linear1 = nn.Linear(linear_channels, linear_channels)

        self.HGT_layer1 = HGTConv(in_channels=linear_channels, out_channels=linear_channels,
                                  metadata=(['user'], [('user', 'follower', 'user'), ('user', 'following', 'user')]))
        self.HGT_layer2 = HGTConv(in_channels=linear_channels, out_channels=linear_channels,
                                  metadata=(['user'], [('user', 'follower', 'user'), ('user', 'following', 'user')]))
        self.gcn1 = GCNConv(linear_channels, out_channel)#测试GCN层



        self.out1 = torch.nn.Linear(out_channel, 64)
        self.out2 = torch.nn.Linear(64, 2)

        self.drop = nn.Dropout(dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()

    def forward(self, des_features, tweet_features, prop_features, cat_features, edge_index, edge_type):
        following_edge_index = edge_index[:, edge_type == 0]
        follower_edge_index = edge_index[:, edge_type == 1]
        # print(follower_edge_index.shape,following_edge_index.shape)
        user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(prop_features)))
        user_features_bool = self.drop(self.ReLU(self.in_linear_bool(cat_features)))
        user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(tweet_features)))
        user_features_des = self.drop(self.ReLU(self.in_linear_des(des_features)))

        user_features = torch.cat((user_features_numeric, user_features_bool, user_features_tweet, user_features_des),
                                  dim=1)
        user_features = self.drop(self.ReLU(self.linear1(user_features)))

        x_dict = {"user": user_features}
        edge_index_dict = {('user', 'follower', 'user'): follower_edge_index,
                           ('user', 'following', 'user'): following_edge_index}

        #原来是两层
        # x_dict = self.HGT_layer1(x_dict, edge_index_dict)
        # x_dict = self.HGT_layer2(x_dict, edge_index_dict)

        #测试一层
        # x_dict = self.HGT_layer2(x_dict, edge_index_dict)

        #测试gcn
        user_features = self.gcn1(user_features,edge_index)

        user_features = self.drop(self.ReLU(self.out1(x_dict["user"])))
        pred = self.out2(user_features)
        # pred_binary = torch.argmax(pred, dim=1)
        return pred


class SHGNDetector(nn.Module):
    def __init__(self, numeric_num=5, cat_num=1, linear_channels=128, tweet_channel=768, des_channel=768, out_channel=128, rel_dim=100, beta=0.05,
                 dropout=0.5):
        super(SHGNDetector, self).__init__()

        self.in_linear_numeric = nn.Linear(numeric_num, int(linear_channels / 4), bias=True)
        self.in_linear_bool = nn.Linear(cat_num, int(linear_channels / 4), bias=True)
        self.in_linear_tweet = nn.Linear(tweet_channel, int(linear_channels / 4), bias=True)
        self.in_linear_des = nn.Linear(des_channel, int(linear_channels / 4), bias=True)
        self.linear1 = nn.Linear(linear_channels, linear_channels)

        # self.HGN_layer1 = SimpleHGN(num_edge_type=2, in_channels=linear_channels, out_channels=out_channel, rel_dim=rel_dim, beta=beta)
        # self.HGN_layer2 = SimpleHGN(num_edge_type=2, in_channels=linear_channels, out_channels=out_channel, rel_dim=rel_dim, beta=beta, final_layer=True)

        self.HGN_layer1 = SimpleHGN(num_edge_type=2, in_channels=linear_channels, out_channels=out_channel,rel_dim=rel_dim, beta=beta)
        self.HGN_layer2 = SimpleHGN(num_edge_type=2, in_channels=linear_channels, out_channels=out_channel,rel_dim=rel_dim, beta=beta)

        self.HGN_layer3 = SimpleHGN(num_edge_type=2, in_channels=linear_channels, out_channels=out_channel,rel_dim=rel_dim, beta=beta, final_layer=True)
        self.gcn1 = GCNConv(linear_channels, out_channel)#测试GCN层



        self.out1 = torch.nn.Linear(out_channel, 64)
        self.out2 = torch.nn.Linear(64, 2)

        self.drop = nn.Dropout(dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()

    def forward(self, des_features, tweet_features, prop_features, cat_features, edge_index, edge_type):
        following_edge_index = edge_index[:, edge_type == 0]
        follower_edge_index = edge_index[:, edge_type == 1]

        user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(prop_features)))
        user_features_bool = self.drop(self.ReLU(self.in_linear_bool(cat_features)))
        user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(tweet_features)))
        user_features_des = self.drop(self.ReLU(self.in_linear_des(des_features)))

        user_features = torch.cat((user_features_numeric, user_features_bool, user_features_tweet, user_features_des),
                                  dim=1)
        user_features = self.drop(self.ReLU(self.linear1(user_features)))

        #这里本来是两层，做对比实验删了，别忘了
        # user_features, alpha = self.HGN_layer1(user_features, edge_index, edge_type)
        # user_features, _ = self.HGN_layer2(user_features, edge_index, edge_type, alpha)

        # user_features, alpha = self.HGN_layer1(user_features, edge_index, edge_type)
        # user_features, _ = self.HGN_layer2(user_features, edge_index, edge_type)
        # user_features, _ = self.HGN_layer3(user_features, edge_index, edge_type)

        #测试gcn
        user_features =self.gcn1(user_features,edge_index)

        user_features = self.drop(self.ReLU(self.out1(user_features)))
        pred = self.out2(user_features)

        return pred

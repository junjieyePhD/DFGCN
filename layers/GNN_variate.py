import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import Parameter
import numpy as np
import math
from layers.Transformer_encoder import TransformerEncoder


class MultiLayerGCN_variate(nn.Module):
    def __init__(self, num_layers, d_model, dropout, n_heads, d_ff, k, activation):
        super(MultiLayerGCN_variate, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1):
            self.layers.append(
                GCN(d_model, d_model, d_model, dropout, n_heads, d_ff, num_layers, activation)
            )
        self.d_model = d_model
        self.k = k


    def person_correlation(self, x):
        # 假设 x 的形状为 batch_size x num_vars x input_len，即 32 x 7 x 96
        batch_size, num_vars, input_len = x.shape

        # 计算均值，形状为 batch_size x num_vars x 1
        mean = x.mean(dim=2, keepdim=True)

        # 中心化数据，形状保持为 batch_size x num_vars x input_len
        centered_data = x - mean

        # 计算协方差矩阵，使用批量矩阵乘法，结果形状为 batch_size x num_vars x num_vars
        cov_matrix = torch.bmm(centered_data, centered_data.transpose(1, 2)) / (input_len - 1)

        # 计算标准差，形状为 batch_size x num_vars
        std_dev = torch.sqrt(torch.diagonal(cov_matrix, dim1=-2, dim2=-1))

        # 防止标准差为0的情况，将0替换为1
        std_dev[std_dev == 0] = 1

        # 计算相关系数矩阵，结果形状为 batch_size x num_vars x num_vars
        std_dev_matrix = std_dev.unsqueeze(2) * std_dev.unsqueeze(1)
        correlation_matrix = cov_matrix / std_dev_matrix

        # 将对角线设为1，确保每个变量与自身的相关系数为1
        # correlation_matrix.fill_diagonal_(1)

        return correlation_matrix

    def edge_index(self, x):
        # 计算每个batch的相似矩阵: [32, 7, 7]
        similarity_matrix = self.person_correlation(x)  # [batch, 7, 7]

        # 直接在CUDA上使用torch的argsort
        k = self.k  # 选择k个最近邻居
        neighbors = torch.argsort(similarity_matrix, dim=-1)[:, :, 1:k + 1]  # [batch, 7, k]

        # 生成行索引 [batch, 7 * k]
        batch_size, num_nodes = similarity_matrix.shape[:2]
        row_indices = torch.arange(num_nodes, device=x.device).repeat(k).reshape(1, -1).repeat(batch_size,
                                                                                            1)  # [batch, 7 * k]

        # 生成列索引 [batch, 7 * k]
        col_indices = neighbors.reshape(batch_size, -1)  # 展平

        # 堆叠生成edge_index [batch, 2, 7 * k]
        edge_index = torch.stack((row_indices, col_indices), dim=1)

        # 确保edge_index在CUDA上
        edge_index = edge_index.long().cuda()

        return edge_index

    def forward(self, enc_out_vari, x_enc):

        edge_index = self.edge_index(x_enc.transpose(2,1))
        # edge_index_all = torch.combinations(torch.arange(x_enc.size(2)), r=2).T.cuda()  # 全连接图  2 * 21
        # 创建每个样本的 Data 对象
        data_list = [Data(x=enc_out_vari[i], edge_index=edge_index[i]) for i in range(enc_out_vari.size(0))]
        # 使用 torch_geometric 的 Batch 机制进行批处理
        batch = Batch.from_data_list(data_list)

        x_raw = batch.x
        edge_index = batch.edge_index

        for layer in self.layers:
            x = layer( enc_out_vari, x_enc, x_raw, edge_index)
            # x = x.reshape(-1, self.d_model)
        return x

# 定义GCN模型
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, n_heads, d_ff, num_layers, activation):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels).cuda()
        self.conv2 = GCNConv(hidden_channels, hidden_channels).cuda()
        self.dropout = nn.Dropout(dropout)
        self.Transformer_encoder_2 = TransformerEncoder(in_channels, n_heads, num_layers, d_ff, dropout).cuda()
        if activation == 'sigmoid':
            self.activate = nn.Sigmoid()
        else:
            self.activate = nn.ReLU()

        self.norm1 = nn.LayerNorm(in_channels)


    def forward(self, enc_out_vari, x_enc, x_raw, edge_index):

        B, M, d_model = enc_out_vari.size()

        x1 = self.conv1(x_raw, edge_index)
        x1 = self.activate(x1)
        x1 = self.dropout(x1)

        x2 = self.conv2(x1, edge_index)
        x2 = self.activate(x2)
        x2 = self.dropout(x2)

        x = x2.reshape(B, M, d_model)

        # ## w/o Attention
        # x = x + enc_out_vari

        enc_out_vari_trans = self.Transformer_encoder_2(enc_out_vari, x, mask=None)    # x作为V，K矩阵，enc_out_vari_embeding作为Q矩阵


        return enc_out_vari_trans



    def Normalization(self, x):

        test_x = torch.matmul(x, x.T)
        means = test_x.mean(1, keepdim=True).detach()
        test_x = test_x - means
        # 计算标准差
        stdev = torch.sqrt(torch.var(test_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # 逐元素相除并将结果赋值回 test_x
        test_x = test_x / stdev

        return test_x

    def visual_jiedian(self,x,edge_index, batch):
        _, a = edge_index.size()
        edge_index = edge_index[:, : a//batch]
        # 可视化图结构
        G = nx.Graph()

        # 添加节点
        G.add_nodes_from(range(7))

        # 添加边
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)
        # 清除以前的图像
        plt.clf()

        # 绘制图
        pos = nx.spring_layout(G)  # 布局
        nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=500, edge_color="black", linewidths=10,
                font_size=15)
        plt.savefig('test.png')
        plt.show()

        # print('ok')


class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.6):
        super(GATModel, self).__init__()

        # 定义第一个 GAT 层
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # 定义第二个 GAT 层，用于输出
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout)

        self.dropout = dropout

    def forward(self, x, edge_index):
        # 第一层 GAT
        x = self.gat1(x, edge_index).sigmoid()
        # x = F.elu(x)  # 激活函数
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第二层 GAT
        x = self.gat2(x, edge_index).sigmoid()
        # x = F.elu(x)  # 激活函数
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        # 定义可学习的权重矩阵 W
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels))

    def forward(self, x, edge_index):
        # Step 1: 添加自环
        num_nodes = x.size(0)
        edge_index_with_self_loops = self.add_self_loops(edge_index, num_nodes)

        # Step 2: 计算度矩阵 D
        row, col = edge_index_with_self_loops
        deg = self.compute_degree(row, num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # 避免分母为0的情况

        # Step 3: 计算 A_hat = D^{-1/2} * A * D^{-1/2}，并构建稀疏矩阵
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        adj = torch.sparse_coo_tensor(edge_index_with_self_loops, norm, (num_nodes, num_nodes))

        # Step 4: 执行图卷积 H^{l+1} = A_hat * X * W
        out = torch.sparse.mm(adj, x)  # 使用稀疏矩阵乘法加速消息传递
        out = torch.matmul(out, self.weight)  # 线性变换
        return out

    def add_self_loops(self, edge_index, num_nodes):
        """为每个节点添加自环"""
        loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index_with_self_loops = torch.cat([edge_index, loop_index], dim=1)
        return edge_index_with_self_loops

    def compute_degree(self, row, num_nodes):
        """计算度矩阵 D"""
        deg = torch.zeros(num_nodes, dtype=torch.float, device=row.device)
        deg.index_add_(0, row, torch.ones(row.size(0), device=row.device))
        return deg

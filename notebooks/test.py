import networkx as nx
import numpy as np
from copy import deepcopy
from sklearn import preprocessing
import random
import torch
import csv
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import torch.optim as optim

# drug_dicts = {}
# with open('/home/bio-17/projects/drug/nf_drug/nfdrug/codes/BFregNN-Cox-for-pyroptosis-in-TNBC/data/9_drug_targets_1.0_revised.tsv') as f:
#     for line in f.readlines():
#         line = line.split('\n')[0].split('\t')
#         drug_dicts[line[0]]=line[1:]

global_graph = json.load(open("/home/bio-17/projects/drug/nf_drug/nfdrug/results/datafetch/res.json", 'r'))
def trans_dicts2graph(graph_dicts):
        G = nx.Graph()
        for key in graph_dicts.keys():
            for k in graph_dicts[key]:
                G.add_edges_from([(key,k)])
        return G
Gs = trans_dicts2graph(graph_dicts = global_graph)


focus_genes_list = []
with open('/home/bio-17/projects/drug/nf_drug/nfdrug/data/gene/ML_gene.csv') as f:
    for index,line in enumerate(f.readlines()):
        if index == 0: continue
        focus_genes_list.append(line.replace("\"","").split()[0])

all_list_gene = pd.read_csv("/home/bio-17/projects/drug/nf_drug/nfdrug/data/gene/Combined_Datasets_Matrix.csv").iloc[:,0].to_list()


drug_dicts = {}
with open('/home/bio-17/projects/drug/nf_drug/nfdrug/data/gene/merged_common_names.csv') as f:
    for index , line in enumerate(f.readlines()):
        if index == 0: continue
        line = line.replace('"','').split(',')
        # drug_dicts[line[0]]=line[1:]
        drug_dicts[line[0]] = list(set([m for m in line[1].strip().split() if m in all_list_gene]))

drug_gene_dicts = drug_dicts
key1 = "Quercetin"
key2 = "Troglitazone"

thres = 2

first_layer_gene = drug_gene_dicts[key1] + drug_gene_dicts[key2]
first_layer_gene.sort()


def shortest_path(G,source,target):
    try:
        num = nx.shortest_path_length(G, source, target)
    except:
        return float('inf')
    return num

def trans_layer(Gs, source_list, target_list):
        affin_graph = []
        removed_target = deepcopy(target_list)
        removed_source = []
        for idx_s, g_s in enumerate(source_list):
            marks_gene = {}
            min_num = float('inf')
            for idx, g_t in enumerate(target_list):
                num = shortest_path(Gs,g_s,g_t)
                if num not in marks_gene:
                    marks_gene[num] = []
                marks_gene[num].append(idx)
                if num > thres:
                    if g_t in removed_target:
                        removed_target.remove(g_t)
                if num < min_num:
                    min_num = num
            if min_num > thres:
                removed_source.append(g_s)
                continue
            for key in marks_gene.keys():
                # print(key,marks_gene[key])
                if key <= thres:
                    for g in marks_gene[key]:
                        affin_graph.append([idx_s,g])
        return removed_source, removed_target, np.array(affin_graph)

removed_s, removed_t, transfer_layer = trans_layer(Gs,
                                                    first_layer_gene,
                                                    focus_genes_list)

def select_subgraph(graph_dicts,gene_list):
        gene_list = list(set(gene_list))
        gene_list.sort()
        return_graph_dicts = {}
        return_graph_dicts['nodes'] = []
        return_graph_dicts['nodes_name'] = []
        return_graph_dicts['edges'] = []
        for idx, g in enumerate(gene_list):
            return_graph_dicts['nodes'].append(idx)
            return_graph_dicts['nodes_name'].append(g)
            if g in graph_dicts:
                for end_nodes in graph_dicts[g]:
                    if end_nodes in gene_list and [idx,gene_list.index(end_nodes)] not in return_graph_dicts['edges']:
                        return_graph_dicts['edges'].append([idx,gene_list.index(end_nodes)])

            return_graph_dicts['edges'].append([idx,idx])
        # return_graph_dicts['edges']=list(set(return_graph_dicts['edges']))
        return return_graph_dicts


graph_dicts = global_graph
second_layer_graph = select_subgraph(graph_dicts, focus_genes_list)

drug_list = drug_gene_dicts[key1] + drug_gene_dicts[key2]
drug_list.sort()

def count_degree(graph_dicts,gene_list):
        degree_list = []
        return_graph_dicts = {}
        return_graph_dicts['edges'] = []
        for idx, g in enumerate(gene_list):
            degree_list.append(0)
            if g in graph_dicts:
                for end_nodes in graph_dicts[g]:
                    if end_nodes in gene_list and \
                    [idx,gene_list.index(end_nodes)] not in return_graph_dicts['edges']:
                        return_graph_dicts['edges'].append([idx,gene_list.index(end_nodes)])
                        degree_list[idx]+=1
        return degree_list

degree_list = count_degree(graph_dicts,drug_list)

drug_list = [d for idx, d in enumerate(drug_list) if not (d in removed_s and degree_list[idx]==0)]

basic_layer_graph = select_subgraph(graph_dicts, drug_list)

first_layer_gene = basic_layer_graph['nodes_name']

def max_connect(basic_layer,second_layer,transfer_layer):
        G = nx.Graph()
        removed = []
        basic_lens = len(basic_layer['nodes'])
        for e in basic_layer['edges']:
            G.add_edge(e[0], e[1])
        for e in second_layer['edges']:
            G.add_edge(e[0] + basic_lens, e[1] + basic_lens)
        for e in transfer_layer:
            G.add_edge(e[0], e[1] + basic_lens)

        largest_graph = nx.connected_components(G)
        nodes = []
        for n in second_layer['nodes']:
            nodes.append(n + basic_lens)
        id_list = []
        for c in largest_graph:
            for n in c:
                if n in nodes:
                    id_list.append(1)
                else:
                    id_list.append(0)
        for idx, c in largest_graph:
            for n in c:
                if id_list[idx] == 0:
                    if n in basic_layer['nodes']:
                        removed.append(basic_layer['nodes_name'][n])
        return removed


removed = max_connect(basic_layer_graph, second_layer_graph, transfer_layer)


removed_s, removed_t, transfer_layer = trans_layer(Gs, first_layer_gene, focus_genes_list)

removed_all = removed_s + removed

drug_list = [d for idx, d in enumerate(drug_list) if not d in removed_all]

basic_layer_graph = select_subgraph(graph_dicts, drug_list)

first_layer_gene = basic_layer_graph['nodes_name']
removed_s, removed_t, transfer_layer = trans_layer(Gs, first_layer_gene, focus_genes_list)

transfer_layer = np.array(transfer_layer).T
basic_layer_adj = np.array(basic_layer_graph['edges']).T
second_layer_adj = np.array(second_layer_graph['edges']).T

csv_reader = csv.reader(open("/home/bio-17/projects/drug/nf_drug/nfdrug/data/gene/Combined_Datasets_Matrix.csv"))
first_line = True
X_index = {}
for line in csv_reader:
        if first_line:
            temp_id_list = line[1:]
            first_line = False
        else:
            gene_name = line[0].strip('\n').strip(' ')
            if gene_name in basic_layer_graph['nodes_name']:
                X_index[gene_name] = line[1:]

X_all = []
for gene_name in basic_layer_graph['nodes_name']:
        if gene_name not in X_index:
            continue
        X_all.append(X_index[gene_name])
X_all=np.array(X_all)
X_all = X_all.transpose()


x_sample_id_list = []
for sample_id in temp_id_list:
        x_sample_id_list.append(sample_id)

csv_reader = csv.reader(open("/home/bio-17/projects/drug/nf_drug/nfdrug/data/gene/Combined_Datasets_Group.csv"))
first_line = True
y_tumor = {} # 1-tumor 0-normal
y_subtypes = {}


p = 1000
for line in csv_reader:
        if first_line:
            first_line = False
        else:
            sample_id = line[0].strip('\n').strip(' ')
            if line[1] == "DKD":
                y_tumor[sample_id] = 1
            elif line[1] == "Control":
                y_tumor[sample_id] = 0

            if line[1] == "DKD":
                y_subtypes[sample_id] = 1
            elif line[1] == "Control":
                y_subtypes[sample_id] = 0
            p += 1

csv_reader = csv.reader(open("/home/bio-17/projects/drug/nf_drug/nfdrug/data/gene/Combined_Datasets_Group.csv"))
first_line = True
y_survival = {}
p = 1000
for line in csv_reader:
        if first_line:
            first_line = False
        else:
            sample_id = line[0].strip('\n').strip(' ')
            if line[1] == "DKD":
                y_survival[sample_id] = (True, p)
            elif line[1] == "Control":
                y_survival[sample_id] = (False, p)
            p += 1

x_data = []
index = []
y_data_label = []
y_data_time = []
for i, sample_id in enumerate(x_sample_id_list):
            x_data.append(X_all[i])
            index.append(sample_id)
            y_data_label.append(y_survival[sample_id][0])
            y_data_time.append(float(y_survival[sample_id][1]))


x_data = np.array(x_data)
y_data_label = np.expand_dims(np.array(y_data_label),axis=0)
y_data_time = np.expand_dims(np.array(y_data_time),axis=0)
y_data = np.concatenate((y_data_label,y_data_time),axis=0).T

def normalize_data(X, y):
    scaler = preprocessing.StandardScaler().fit(X)
    X_transformed = scaler.transform(X)
    return X_transformed, y

X, y = normalize_data(x_data, y_data)


class GeneDataset(Dataset):
    def __init__(self, data, label):
        super(GeneDataset).__init__()
        self.x = data
        self.y = label
    def __getitem__(self, index):
        sample_x = self.x[index,:]
        sample_y = self.y[index,:]
        return [sample_x, sample_y]
    def __len__(self):
        return self.x.shape[0]

train_data_dataset = GeneDataset(X, y)
train_data = DataLoader(train_data_dataset, batch_size=10)

cox_weights = {'EIF4A3':0.041459,'ISG20':-0.030420,
               'MDN1':-0.002452,'RPLP0':-0.850369,
               'TP53':-0.170639}
cox_weights_list = []

for g in second_layer_graph['nodes_name']:
    cox_weights_list.append(cox_weights[g])

cox_weights_list = np.array(cox_weights_list)
cox_weights_list = np.expand_dims(cox_weights_list, axis=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCN(nn.Module):
    def __init__(self, in_dim,  out_dim):
        super(GCN, self).__init__()

        self.W = nn.Parameter(torch.randn(in_dim, out_dim))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))


    def forward(self, x, adj):
        support = torch.matmul(x, self.W)
        output = torch.matmul(adj, support)
        return output


class SGC(nn.Module):
    def __init__(self, in_dim,  out_dim):
        super(SGC, self).__init__()
        if in_dim != out_dim:
            print("SGC dim error")


    def forward(self, x, adj):
        output = torch.matmul(adj, x)
        return output


class IntraLayer(nn.Module):
    def __init__(self, n_nodes, in_dim, hid_dim, out_dim, mode, gnn_mode):
        super(IntraLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if gnn_mode == "GCN":
            self.gnn = GCN(in_dim, hid_dim)
        elif gnn_mode == "SGC":
            self.gnn = SGC(in_dim, hid_dim)


    def forward(self, x, adj):

        out = F.elu(self.gnn(x, adj))

        return out

class InterLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(InterLayer, self).__init__()

        self.W = nn.Parameter(torch.randn(in_features, out_features))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x, adj):

        w = self.W * adj
        w = w.transpose(1, 0)

        x = x.transpose(1, 2)
        x = F.linear(x, w)
        x = x.transpose(1, 2)

        x = self.batch_norm(x)
        x = F.elu(x)

        return x

class cox_affine(nn.Module):
    def __init__(self, in_dim2):
        super().__init__()
        self.gene_num = in_dim2
        self.aff = nn.Linear(4,1)
    def forward(self,x):
        n_samples = x.shape[0]
        x.requires_grad_()
        x = x.reshape(n_samples, self.gene_num, -1)
        x = self.aff(x)
        x = x.squeeze()
        return x



class BFRegNN(nn.Module):
    def __init__(self, in_dim, in_dim2, n_hid, basic_layer, transfer_layer, second_layer):
        super().__init__()

        self.graph1 = basic_layer.to_dense()
        self.basic_graph = IntraLayer(in_dim, 1, 4, 4, "cat", "GCN")

        self.transfer_graph = transfer_layer.to_dense()
        self.inter_layer = InterLayer(in_dim, in_dim2)

        self.graph2 = second_layer.to_dense()
        self.second_graph = IntraLayer(in_dim2, 4, 4, 4, "cat", "GCN")

        self.cox_aff = cox_affine(in_dim2)


    def forward(self, x):
        print("第一步输入：",x.shape)
        x = x.unsqueeze(-1)
        print("第一步输入增加：",x.shape)
        x = self.basic_graph(x, self.graph1)
        print("第二步输出：",x.shape)
        x = self.inter_layer(x, self.transfer_graph)
        print("第三步输出：",x.shape)
        x = self.second_graph(x, self.graph2)
        print("第四步输出：",x.shape)
        x = self.cox_aff(x)
        print("第五步输出：",x.shape)

        return x

# 这段代码定义了一个名为 `cox_module` 的 PyTorch 模块，
# 用于实现 Cox 比例风险模型的负对数似然损失和一致性指数（Concordance Index）的计算。
# 以下是对代码各部分的详细解释：

# 类定义和初始化
# `cox_module` 继承自 `nn.Module`，是一个自定义的 PyTorch 模块。
class cox_module(nn.Module):
    # 初始化方法，接收输入维度 `in_dim2` 和 Cox 权重列表 `cox_weights_list` 作为参数。
    def __init__(self, in_dim2, cox_weights_list):
        # 调用父类的初始化方法。
        super().__init__()
        # 保存输入维度作为基因数量。
        self.gene_num = in_dim2
        # 将 Cox 权重列表转换为 PyTorch 张量，并设置为不可训练（`requires_grad=False`）。
        self.W = torch.tensor(cox_weights_list, requires_grad=False).float()

    # 前向传播方法，计算负对数似然损失和一致性指数。
    def forward(self, x, event, time, alpha=0, beta=0):
        # 按时间降序排序，返回排序后的索引 `o`。
        _, o = torch.sort(-time, dim=0, stable=True)
        # 根据排序索引对事件标签进行排序。
        my_event = event[o]
        # 根据排序索引对输入特征进行排序。
        x = x[o,:]
        # 根据排序索引对时间进行排序。
        my_time = time[o]
        # 初始化损失为 0。
        loss = 0
        # 计算输入特征与 Cox 权重的矩阵乘法。
        xw = torch.matmul(x, self.W.to(x.device))
        # 调用 `neg_par_log_likelihood` 方法计算负对数似然损失、风险集总和、差异和预测值。
        loss, risksets, diff, pred = self.neg_par_log_likelihood(xw, my_time, my_event)
        # 计算损失的均值。
        loss = loss.mean()
        # 调用 `c_index` 方法计算一致性指数。
        self.concordance = self.c_index(xw, my_time, my_event)
        # 返回损失的扩展维度和其他相关信息。
        return loss.unsqueeze(-1),(risksets, diff, pred, my_time, my_event)

    # 计算负对数似然损失的方法。
    def neg_par_log_likelihood(self,pred, ytime, yevent):
        # 计算观察到的事件数量，并添加一个小的常数以避免除零错误。
        n_observed = yevent.sum(0) + 1e-6
        # 调用 `R_set` 方法生成时间指示矩阵。
        ytime_indicator = self.R_set(ytime)
        # 计算风险集总和。
        risk_set_sum = ytime_indicator.mm(torch.exp(pred))
        # 计算预测值与风险集总和对数的差异。
        diff = pred - torch.log(risk_set_sum)
        # 扩展事件标签的维度。
        yevent = yevent.unsqueeze(-1)
        # 计算观察到的事件中差异的总和。
        sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)
        # 计算负对数似然损失。
        cost = ( -(sum_diff_in_observed / n_observed)).reshape((-1,))
        # 返回损失、风险集总和、差异和预测值。
        return(cost,risk_set_sum,diff,pred)

    # 计算一致性指数的方法。
    def c_index(self, pred, ytime, yevent):
        # 获取样本数量。
        n_sample = len(ytime)
        # 调用 `R_set` 方法生成时间指示矩阵。
        ytime_indicator = self.R_set(ytime)
        # 移除时间指示矩阵的对角线元素。
        ytime_matrix = ytime_indicator - torch.diag(torch.diag(ytime_indicator))
        # 找到未发生事件（被删失）的样本索引。
        censor_idx = (yevent == 0).nonzero()
        # 创建一个全零张量。
        zeros = torch.zeros(n_sample).to(ytime_matrix.device)
        # 将被删失样本对应的行设置为零。
        ytime_matrix[censor_idx, :] = zeros
        # 创建一个与时间矩阵形状相同的全零矩阵。
        pred_matrix = torch.zeros_like(ytime_matrix)
        # 计算预测值之间的差异。
        pred_diffs = pred - pred.T
        # 如果预测值差异大于 0，则将预测矩阵对应位置设置为 1。
        pred_matrix[pred_diffs > 0] = 1
        # 如果预测值差异等于 0，则将预测矩阵对应位置设置为 0.5。
        pred_matrix[pred_diffs == 0] = 0.5
        # 计算一致性矩阵。
        concord_matrix = pred_matrix.mul(ytime_matrix)
        # 计算一致性矩阵的总和。
        concord = torch.sum(concord_matrix)
        # 计算分母，添加一个小的常数以避免除零错误。
        epsilon = torch.sum(ytime_matrix) + 1e-6
        # 计算一致性指数。
        concordance_index = torch.div(concord, epsilon)
        # 返回一致性指数。
        return concordance_index

    # 生成时间指示矩阵的方法。
    def R_set(self,x):
        # 获取样本数量。
        n_sample = x.shape[0]
        # 创建一个全 1 矩阵。
        matrix_ones = torch.ones(n_sample, n_sample).to(x.device)
        # 生成下三角矩阵作为时间指示矩阵。
        indicator_matrix = torch.tril(matrix_ones).to(x.device)
        # 返回时间指示矩阵。
        return(indicator_matrix)


class BFRegNN_COX(nn.Module):
    def __init__(self, in_dim, in_dim2, n_hid, graphs1, transfer_layer, second_layer, cox_weights_list):
        super().__init__()
        self.bfregNN = BFRegNN(in_dim, in_dim2, n_hid, graphs1, transfer_layer, second_layer)
        # self.neg_module = cox_module(in_dim2, cox_weights_list)
        self.sig  = nn.Sigmoid()
        self.flat = nn.Flatten()
        self.linear = nn.Linear(in_dim2,1)


    def forward(self, x, event, time):
        x = self.bfregNN(x)
        x = self.sig(self.flat(self.linear(x)).squeeze(-1))
        print(x)
        print("x 的sigmoid" + str(x.shape))
        # loss = self.neg_module(x, event, time)
        # self.concordance = self.neg_module.concordance

        return x


def max_indice_collapsed(ori_gene):
    max_index = ori_gene.size(0) - 1
    indices = ori_gene.coalesce().indices()
    mask = indices >= max_index
    if torch.any(mask):
        # print("发现超出范围的索引：")
        # print(indices[:, mask.any(0)])  # 打印超出范围的索引

        # 修正超出范围的索引（示例：将超出范围的索引截断为最大索引）
        indices[mask] = max_index - 1
        ori_gene = torch.sparse_coo_tensor(indices, ori_gene.coalesce().values(), ori_gene.size())

        # 再次检查
        mask = indices >= max_index
        if torch.any(mask):
            print("修正后仍有超出范围的索引！")
        else:
            pass
    print(ori_gene.shape)
    # print(ori_gene.to_dense())
    return ori_gene


def build_bfregNN_model(gene_num, gene_num2, gene_adj, gene_adj2, transfer_layer, device, cox_weights_list):

    # device = torch.device("cpu")

    v1 = torch.ones(gene_adj.shape[1], device=device)
    print(v1.shape)
    ori_gene = torch.sparse_coo_tensor(gene_adj, v1, size=(gene_num, gene_num))

    # ori_gene = max_indice_collapsed(ori_gene)
    # max_index = ori_gene.size(0) - 1
    # indices = ori_gene.coalesce().indices()
    # mask = indices >= max_index
    # if torch.any(mask):
    #     print("发现超出范围的索引：")
    #     print(indices[:, mask.any(0)])  # 打印超出范围的索引

    #     # 修正超出范围的索引（示例：将超出范围的索引截断为最大索引）
    #     indices[mask] = max_index - 1
    #     ori_gene = torch.sparse_coo_tensor(indices, ori_gene.coalesce().values(), ori_gene.size())

    #     # 再次检查
    #     mask = indices >= max_index
    #     if torch.any(mask):
    #         print("修正后仍有超出范围的索引！")
    #     else:
    #         print("索引已修正。")

    # print(ori_gene.shape)
    # print(ori_gene.to_dense())
    # max_index = ori_gene.size(0) - 1
    # indices = ori_gene.coalesce().indices()
    # if torch.any(indices >= max_index):
    #     print("索引超出范围！")
    #     print(max_index)
    #     print(indices)
    # else:
    #     print("索引在范围内。")

    # try:
    #     dense_tensor = ori_gene.to_dense()
    #     print(dense_tensor)
    # except RuntimeError as e:
    #     print(f"转换密集张量时发生错误：{e}")

    # print(ori_gene.shape)
    # print(ori_gene.to_dense())

    v2 = torch.ones(gene_adj2.shape[1], device=device)
    ori_gene2 = torch.sparse_coo_tensor(gene_adj2, v2, size=(gene_num2, gene_num2))
    # ori_gene2 = max_indice_collapsed(ori_gene2).to_dense()
    print(v2)
    print(ori_gene2)

    v3 = torch.ones(transfer_layer.shape[1], device=device)
    transfer_layer = torch.sparse_coo_tensor(transfer_layer, v3, size=(gene_num,gene_num2)).to_dense()


    model = BFRegNN_COX(gene_num, gene_num2, 64, ori_gene, transfer_layer, ori_gene2, cox_weights_list).to(device)

    return model

model = build_bfregNN_model(X.shape[1], cox_weights_list.shape[0],
                            basic_layer_adj, second_layer_adj, transfer_layer,
                              device, cox_weights_list)

print(model)
optimizer = optim.Adam(model.parameters(), lr =1e-4, weight_decay = 1e-4)
cross = nn.CrossEntropyLoss()

def calculate_accuracy(pred, event_label):
    """
    计算准确率
    :param pred: 模型输出的预测值（范围在 0 到 1 之间）
    :param event_label: 真实标签（0 或 1）
    :return: 准确率
    """
    # 将 pred 中的值转换为 0 或 1
    pred_labels = (pred >= 0.5).float()  # 大于等于 0.5 为 1，否则为 0

    # 计算预测正确的样本数
    correct = (pred_labels == event_label).sum().item()

    # 计算准确率
    accuracy = correct / event_label.size(0)

    return accuracy

def train_model(model, train_data, optimizer, epoch, device):
    patience = 500
    patience_count = 0
    global_loss = 0
    global_con = 0

    losses = []  # 保存每个 epoch 的损失
    accuracies = []  # 保存每个 epoch 的准确率

    for e in range(epoch):
        train_loss = 0
        total_correct = 0
        total_samples = 0

        for d in train_data:
            optimizer.zero_grad()
            data = d[0].float().to(device)
            label = d[1].float().to(device)
            event_label = label[:, 0]
            time_label = label[:, 1]

            pred = model(data, event_label, time_label)

            # 计算损失
            loss = cross(pred, event_label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 计算准确率
            pred_labels = (pred >= 0.5).float()  # 将预测值转换为 0 或 1
            correct = (pred_labels == event_label).sum().item()
            total_correct += correct
            total_samples += event_label.size(0)

        # 计算平均损失和准确率
        avg_loss = train_loss / len(train_data)
        avg_accuracy = total_correct / total_samples

        # 保存当前 epoch 的损失和准确率
        losses.append(avg_loss)
        accuracies.append(avg_accuracy)

        print(f"Epoch {e + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    return losses, accuracies
# def train_model(model, train_data, optimizer, epoch, device):
#     patience = 500
#     patience_count = 0
#     global_loss = 0
#     global_con = 0

#     losses = []  # 保存每个 epoch 的损失
#     accuracies = []  # 保存每个 epoch 的准确率

#     for e in range(epoch):
#         train_loss = 0
#         correct = 0
#         total = 0

#         for d in train_data:
#             optimizer.zero_grad()
#             data = d[0].float().to(device)
#             label = d[1].float().to(device)
#             event_label = label[:, 0]
#             time_label = label[:, 1]

#             pred = model(data, event_label, time_label)

#             # 计算损失
#             loss = cross(pred, event_label)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()

#             # 计算准确率
#             accuracy = calculate_accuracy(pred, event_label)
#             accuracies.append(accuracy)

#         # 计算平均损失
#         train_loss /= len(train_data)
#         losses.append(train_loss)

#         # accuracies.append(accuracy)

#         print(f"Epoch {e + 1}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}")

#     return losses, accuracies

# def train_model(model, train_data, optimizer, epoch, device):
#     patience = 500
#     patience_count = 0
#     global_loss = 0
#     global_con = 0

#     for e in range(epoch):
#         train_loss = 0
#         concordance = 0
#         for d in train_data:
#             optimizer.zero_grad()
#             data = d[0].float().to(device)
#             print(data.shape)
#             label = d[1].float().to(device)
#             event_label = label[:,0]
#             print(event_label.shape)
#             time_label = label[:,1]

#             pred = model(data, event_label, time_label)
#             print("pred" + str(pred))
#             print("event_label" + str(event_label))

#             correct = (pred.round() == event_label).sum().item()
#             accuracy = correct / event_label.size(0)
#             loss = cross(pred, event_label)
#             print(loss)

#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()



#         if e == 0:
#             losses = []
#             accuracies = []

#         losses.append(train_loss)
#         accuracies.append(accuracy)



#     return losses,accuracies#global_loss.item(), global_con.item()

loss,acc = train_model(model, train_data, optimizer, 2000, device)


import matplotlib.pyplot as plt

loss = [round(float(i), 2) for i in loss]
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(loss, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(acc, label="acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.legend()


plt.tight_layout()
plt.show()

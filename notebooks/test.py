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

drug_dicts = {}
with open('/home/bio-17/projects/drug/nf_drug/nfdrug/codes/BFregNN-Cox-for-pyroptosis-in-TNBC/data/9_drug_targets_1.0_revised.tsv') as f:
    for line in f.readlines():
        line = line.split('\n')[0].split('\t')
        drug_dicts[line[0]]=line[1:]

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


class BFRegNN_COX(nn.Module):
    def __init__(self, in_dim, in_dim2, n_hid, graphs1, transfer_layer, second_layer, cox_weights_list):
        super().__init__()
        self.bfregNN = BFRegNN(in_dim, in_dim2, n_hid, graphs1, transfer_layer, second_layer)
        # self.neg_module = cox_module(in_dim2, cox_weights_list)

    def forward(self, x, event, time):
        x = self.bfregNN(x)
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
optimizer = optim.Adam(model.parameters(), lr =1e-1, weight_decay = 1e-4)

def train_model(model, train_data, optimizer, epoch, device):
    patience = 500
    patience_count = 0
    global_loss = 0
    global_con = 0

    for e in range(epoch):
        train_loss = 0
        concordance = 0
        for d in train_data:
            optimizer.zero_grad()
            data = d[0].float().to(device)
            print(data.shape)
            label = d[1].float().to(device)
            event_label = label[:,0]
            time_label = label[:,1]

            loss = model(data, event_label, time_label)
            print(loss.shape)
            # ttttt
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()

            train_loss += loss
            concordance += model.concordance

        train_loss /= len(train_data)
        concordance /= len(train_data)

        print(e, concordance.item(), train_loss.item())

        if concordance > global_con:
            global_con = concordance
            global_loss = train_loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count == patience:
                break

    return global_loss.item(), global_con.item()

loss, con_loss = train_model(model, train_data, optimizer, 200, device)

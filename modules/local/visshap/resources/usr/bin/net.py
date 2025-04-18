#!/usr/bin/env python

from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

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
        x = x.unsqueeze(-1)
        x = self.basic_graph(x, self.graph1)
        x = self.inter_layer(x, self.transfer_graph)
        x = self.second_graph(x, self.graph2)
        x = self.cox_aff(x)
        return x

class BFRegNN_COX(nn.Module):
    def __init__(self, in_dim, in_dim2, n_hid, graphs1, transfer_layer, second_layer):
        super().__init__()
        self.bfregNN = BFRegNN(in_dim, in_dim2, n_hid, graphs1, transfer_layer, second_layer)
        self.sig  = nn.Sigmoid()
        self.flat = nn.Flatten()
        self.linear = nn.Linear(in_dim2,1)

    def forward(self, x, event, time):
        x = self.bfregNN(x)
        index = torch.argmax(x, dim=-1)
        if x.dim() > 1:
            x = self.sig(self.flat(self.linear(x)).squeeze(-1))
        else:
            x = self.sig(self.flat(self.linear(x.unsqueeze(0))).squeeze(-1))
        return x,index



class BuildModel(object):
    def build_bfregNN_model(gene_num, gene_num2, gene_adj, gene_adj2,
                            transfer_layer, device, cox_weights_list):
        v1 = torch.ones(gene_adj.shape[1], device=device)
        ori_gene = torch.sparse_coo_tensor(gene_adj, v1, size=(gene_num, gene_num))

        v2 = torch.ones(gene_adj2.shape[1], device=device)
        ori_gene2 = torch.sparse_coo_tensor(gene_adj2, v2, size=(gene_num2, gene_num2))

        v3 = torch.ones(transfer_layer.shape[1], device=device)
        transfer_layer = torch.sparse_coo_tensor(transfer_layer, v3, size=(gene_num,gene_num2)).to_dense()

        model = BFRegNN_COX(gene_num, gene_num2, 64, ori_gene, transfer_layer, ori_gene2, cox_weights_list).to(device)
        return model

#!/usr/bin/env python

import argparse
import numpy as np
import networkx as nx
import numpy as np
from copy import deepcopy
from sklearn import preprocessing
import random
import csv
import json


def generate_network_architecture(graph_dicts, drug_gene_dicts, key1, key2, thres):

    def count_degree(graph_dicts,gene_list):
        degree_list = []
        return_graph_dicts = {}
        return_graph_dicts['edges'] = []
        for idx, g in enumerate(gene_list):
            degree_list.append(0)
            if g in graph_dicts:
                for end_nodes in graph_dicts[g]:
                    if end_nodes in gene_list and [idx,gene_list.index(end_nodes)] not in return_graph_dicts['edges']:
                        return_graph_dicts['edges'].append([idx,gene_list.index(end_nodes)])
                        degree_list[idx]+=1
        return degree_list


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


    def trans_dicts2graph(graph_dicts):
        G = nx.Graph()
        for key in graph_dicts.keys():
            for k in graph_dicts[key]:
                G.add_edges_from([(key,k)])
        return G


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


    Gs = trans_dicts2graph(graph_dicts)

    focus_genes_list = []
    with open('/home/bio-17/projects/drug/nf_drug/nfdrug/codes/BFregNN-Cox-for-pyroptosis-in-TNBC/data/gene_pyroptosis_9.txt') as f:
        for line in f.readlines():
            focus_genes_list.append(line.split()[0])

    first_layer_gene = drug_gene_dicts[key1] + drug_gene_dicts[key2]
    first_layer_gene.sort()
    removed_s, removed_t, transfer_layer = trans_layer(Gs, first_layer_gene, focus_genes_list)
    second_layer_graph = select_subgraph(graph_dicts, focus_genes_list)


    drug_list = drug_gene_dicts[key1] + drug_gene_dicts[key2]
    drug_list.sort()

    degree_list = count_degree(graph_dicts,drug_list)
    drug_list = [d for idx, d in enumerate(drug_list) if not (d in removed_s and degree_list[idx]==0)]


    basic_layer_graph = select_subgraph(graph_dicts, drug_list)
    first_layer_gene = basic_layer_graph['nodes_name']
    removed = max_connect(basic_layer_graph, second_layer_graph, transfer_layer)
    removed_s, removed_t, transfer_layer = trans_layer(Gs, first_layer_gene, focus_genes_list)
    removed_all = removed_s + removed


    drug_list = [d for idx, d in enumerate(drug_list) if not d in removed_all]
    basic_layer_graph = select_subgraph(graph_dicts, drug_list)
    first_layer_gene = basic_layer_graph['nodes_name']
    removed_s, removed_t, transfer_layer = trans_layer(Gs, first_layer_gene, focus_genes_list)

    return basic_layer_graph, transfer_layer, second_layer_graph




drug_dicts = {}
with open('/home/bio-17/projects/drug/nf_drug/nfdrug/data/gene/merged_common_names.csv') as f:
    for index , line in enumerate(f.readlines()):
        if index == 0: continue
        line = line.replace('"','').split(',')
        # drug_dicts[line[0]]=line[1:]
        drug_dicts[line[0]] = line[1].strip().split()  
print(drug_dicts)

global_graph = json.load(open("/home/bio-17/projects/drug/nf_drug/nfdrug/results/datafetch/res.json", 'r'))
basic_layer, transfer_layer, second_layer = \
    generate_network_architecture(global_graph, drug_dicts,\
    "Quercetin","Troglitazone", 2)
transfer_layer = np.array(transfer_layer).T
basic_layer_adj = np.array(basic_layer['edges']).T
second_layer_adj = np.array(second_layer['edges']).T

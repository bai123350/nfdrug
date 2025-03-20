#!/usr/bin/env python

import argparse
import numpy as np
import networkx as nx
from copy import deepcopy

# import random
# import csv
import json
import pandas as pd


def trans_dicts2graph(graph_dicts: dict) -> nx.Graph:
    """
    将字典形式的图数据转换为NetworkX图对象。

    参数:
    graph_dicts (dict): 字典形式的图数据，其中键表示节点，值表示与该节点相连的节点列表。

    返回值:
    G (nx.Graph): 转换后的NetworkX图对象，包含所有节点和边。
    """
    # 创建一个空的NetworkX图对象
    G = nx.Graph()

    # 遍历字典中的每个节点及其相连节点列表
    for key in graph_dicts.keys():
        for k in graph_dicts[key]:
            # 将节点对作为边添加到图中
            G.add_edges_from([(key, k)])

    # 返回构建好的图对象
    return G


def shortest_path(G: nx.Graph, source: str, target: str) -> float:
    """
    计算图中从源节点到目标节点的最短路径长度。

    参数:
    G (networkx.Graph): 输入的图对象，表示需要计算最短路径的图。
    source (node): 源节点，表示路径的起点。
    target (node): 目标节点，表示路径的终点。

    返回值:
    int or float: 如果存在从源节点到目标节点的路径，则返回最短路径的长度；
                  如果不存在路径，则返回无穷大（float('inf')）。
    """
    try:
        # 使用networkx库的shortest_path_length函数计算最短路径长度
        num = nx.shortest_path_length(G, source, target)
    except:
        # 如果计算过程中发生异常（例如节点之间不存在路径），返回无穷大
        return float("inf")
    return num


def trans_layer(
    Gs: nx.Graph, source_list: list, target_list: list, thres: int = 2
) -> tuple:
    """
    该函数用于在给定的图中，根据源节点列表和目标节点列表，计算它们之间的最短路径，并根据阈值筛选出符合条件的节点对。

    参数:
    - Gs: 图对象，表示包含节点和边的图结构。
    - source_list: 列表，包含源节点的标识符。
    - target_list: 列表，包含目标节点的标识符。
    - thres: 整数，表示最短路径的阈值，默认为2。

    返回值:
    - removed_source: 列表，包含被移除的源节点，这些节点与所有目标节点的最短路径都大于阈值。
    - removed_target: 列表，包含被移除的目标节点，这些节点与所有源节点的最短路径都大于阈值。
    - affin_graph: 二维数组，表示符合条件的源节点和目标节点的索引对。
    """
    affin_graph = []
    removed_target = deepcopy(target_list)
    removed_source = []

    # 遍历源节点列表，计算每个源节点与所有目标节点的最短路径
    for idx_s, g_s in enumerate(source_list):
        marks_gene = {}
        min_num = float("inf")

        # 遍历目标节点列表，计算最短路径并记录相关信息
        for idx, g_t in enumerate(target_list):
            num = shortest_path(Gs, g_s, g_t)
            if num not in marks_gene:
                marks_gene[num] = []
            marks_gene[num].append(idx)

            # 如果最短路径大于阈值，则从removed_target中移除该目标节点
            if num > thres:
                if g_t in removed_target:
                    removed_target.remove(g_t)

            # 更新当前源节点与所有目标节点的最短路径的最小值
            if num < min_num:
                min_num = num

        # 如果当前源节点与所有目标节点的最短路径都大于阈值，则将其添加到removed_source中
        if min_num > thres:
            removed_source.append(g_s)
            continue

        # 将符合条件（最短路径小于等于阈值）的源节点和目标节点的索引对添加到affin_graph中
        for key in marks_gene.keys():
            if key <= thres:
                for g in marks_gene[key]:
                    affin_graph.append([idx_s, g])

    return removed_source, removed_target, np.array(affin_graph)


def select_subgraph(graph_dicts: dict, gene_list: list) -> dict:
    """
    从给定的图字典中提取与基因列表相关的子图。

    参数:
    graph_dicts (dict): 表示图的字典，键为基因名称，值为与该基因相连的其他基因列表。
    gene_list (list): 包含需要提取的基因名称的列表。

    返回值:
    dict: 返回一个包含子图信息的字典，包含以下键：
        - 'nodes': 子图中节点的索引列表。
        - 'nodes_name': 子图中节点的名称列表。
        - 'edges': 子图中边的列表，每条边由两个节点的索引表示。
    """
    # 去重并排序基因列表
    gene_list = list(set(gene_list))
    gene_list.sort()

    # 初始化返回的子图字典
    return_graph_dicts = {}
    return_graph_dicts["nodes"] = []
    return_graph_dicts["nodes_name"] = []
    return_graph_dicts["edges"] = []

    # 遍历基因列表，构建子图的节点和边
    for idx, g in enumerate(gene_list):
        return_graph_dicts["nodes"].append(idx)
        return_graph_dicts["nodes_name"].append(g)

        # 如果当前基因在图字典中，则添加与之相连的边
        if g in graph_dicts:
            for end_nodes in graph_dicts[g]:
                if (
                    end_nodes in gene_list
                    and [idx, gene_list.index(end_nodes)]
                    not in return_graph_dicts["edges"]
                ):
                    return_graph_dicts["edges"].append(
                        [idx, gene_list.index(end_nodes)]
                    )

        # 添加自环边
        return_graph_dicts["edges"].append([idx, idx])

    return return_graph_dicts


def count_degree(graph_dicts, gene_list):
    degree_list = []
    return_graph_dicts = {}
    return_graph_dicts["edges"] = []
    for idx, g in enumerate(gene_list):
        degree_list.append(0)
        if g in graph_dicts:
            for end_nodes in graph_dicts[g]:
                if (
                    end_nodes in gene_list
                    and [idx, gene_list.index(end_nodes)]
                    not in return_graph_dicts["edges"]
                ):
                    return_graph_dicts["edges"].append(
                        [idx, gene_list.index(end_nodes)]
                    )
                    degree_list[idx] += 1
    return degree_list

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="data/gene/merged_common_names.csv",
    )
    parser.add_argument(
        "--path1", type=str, default="data/gene/Combined_Datasets_Matrix.csv"
    )
    parser.add_argument("--path2", type=str, default="data/gene/ML_gene.csv")
    parser.add_argument("--json", type=str, default="res.json")
    parser.add_argument("--drug1", type=str, default="mitoxantrone")
    parser.add_argument("--drug2", type=str, default="gambogic acid")
    parser.add_argument("--out", type=str, default="output")
    args = parser.parse_args()

    all_list_gene = pd.read_csv(args.path1).iloc[:, 0].to_list()
    global_graph = json.load(open(args.json, "r"))
    Gs = trans_dicts2graph(graph_dicts=global_graph)

    focus_genes_list = []
    with open(args.path2) as f:
        for index, line in enumerate(f.readlines()):
            if index == 0:
                continue
            focus_genes_list.append(line.replace('"', "").split()[0])

    drug_dicts = {}
    with open(args.path) as f:
        for index, line in enumerate(f.readlines()):
            if index == 0:
                continue
            line = line.replace('"', "").split(",")
            drug_dicts[line[0]] = list(
                set([m for m in line[1].strip().split() if m in all_list_gene])
            )
    json.dump(drug_dicts, open(f"{args.out}_drug.json", "w"))

    first_layer_gene = drug_dicts[args.drug1] + drug_dicts[args.drug2]
    first_layer_gene.sort()

    removed_s, removed_t, transfer_layer = trans_layer(
        Gs, first_layer_gene, focus_genes_list
    )

    second_layer_graph = select_subgraph(global_graph, focus_genes_list)

    drug_list = drug_dicts[args.drug1] + drug_dicts[args.drug2]
    drug_list.sort()

    degree_list = count_degree(global_graph, drug_list)

    drug_list = [d for idx, d in enumerate(drug_list) if not (d in removed_s and degree_list[idx]==0)]

    basic_layer_graph = select_subgraph(global_graph, drug_list)

    first_layer_gene = basic_layer_graph['nodes_name']

    removed = max_connect(basic_layer_graph, second_layer_graph, transfer_layer)

    removed_s, removed_t, transfer_layer = trans_layer(Gs, first_layer_gene, focus_genes_list)

    removed_all = removed_s + removed

    drug_list = [d for idx, d in enumerate(drug_list) if not d in removed_all]

    basic_layer_graph = select_subgraph(global_graph, drug_list)

    first_layer_gene = basic_layer_graph['nodes_name']
    removed_s, removed_t, transfer_layer = trans_layer(Gs, first_layer_gene, focus_genes_list)

    transfer_layer = np.array(transfer_layer).T
    basic_layer_adj = np.array(basic_layer_graph['edges']).T
    second_layer_adj = np.array(second_layer_graph['edges']).T

    np.savez(f"{args.out}_transfer.npz", trans_layer=transfer_layer)





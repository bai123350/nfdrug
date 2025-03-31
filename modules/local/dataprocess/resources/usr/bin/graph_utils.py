import networkx as nx
import numpy as np
from copy import deepcopy

class GraphUtils:
    @staticmethod
    def trans_dicts2graph(graph_dicts: dict) -> nx.Graph:
        """
        将字典形式的图数据转换为NetworkX图对象。

        参数:
        graph_dicts (dict): 字典形式的图数据，其中键表示节点，值表示与该节点相连的节点列表。

        返回值:
        G (nx.Graph): 转换后的NetworkX图对象，包含所有节点和边。
        """
        G = nx.Graph()
        for key in graph_dicts.keys():
            for k in graph_dicts[key]:
                G.add_edges_from([(key, k)])
        return G

    @staticmethod
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
            num = nx.shortest_path_length(G, source, target)
        except:
            return float("inf")
        return num

    @staticmethod
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
        gene_list = list(set(gene_list))
        gene_list.sort()

        return_graph_dicts = {}
        return_graph_dicts["nodes"] = []
        return_graph_dicts["nodes_name"] = []
        return_graph_dicts["edges"] = []

        for idx, g in enumerate(gene_list):
            return_graph_dicts["nodes"].append(idx)
            return_graph_dicts["nodes_name"].append(g)

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
            return_graph_dicts["edges"].append([idx, idx])

        return return_graph_dicts

    @staticmethod
    def count_degree(graph_dicts: dict, gene_list: list) -> list:
        """
        计算基因列表中每个基因的度（degree），并返回一个包含度的列表。

        参数:
        graph_dicts (dict): 一个字典，表示图的邻接表。键是基因，值是与该基因相连的基因列表。
        gene_list (list): 一个基因列表，表示需要计算度的基因。

        返回值:
        list: 一个包含每个基因度的列表，顺序与gene_list中的基因顺序一致。
        """
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

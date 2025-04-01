#!/usr/bin/env python

import numpy as np
from copy import deepcopy
import networkx as nx

class LayerUtils:
    @staticmethod
    def trans_layer(Gs: nx.Graph, source_list: list, target_list: list, thres: int = 2) -> tuple:
        affin_graph = []
        removed_target = deepcopy(target_list)
        removed_source = []

        for idx_s, g_s in enumerate(source_list):
            marks_gene = {}
            min_num = float("inf")

            for idx, g_t in enumerate(target_list):
                num = LayerUtils.shortest_path(Gs, g_s, g_t)
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
                if key <= thres:
                    for g in marks_gene[key]:
                        affin_graph.append([idx_s, g])

        return removed_source, removed_target, np.array(affin_graph)

    @staticmethod
    def max_connect(basic_layer, second_layer, transfer_layer):
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
        nodes = [n + basic_lens for n in second_layer['nodes']]
        id_list = []

        for c in largest_graph:
            for n in c:
                if n in nodes:
                    id_list.append(1)
                else:
                    id_list.append(0)

        for idx, c in enumerate(largest_graph):
            for n in c:
                if id_list[idx] == 0 and n in basic_layer['nodes']:
                    removed.append(basic_layer['nodes_name'][n])

        return removed

    @staticmethod
    def shortest_path(G: nx.Graph, source: str, target: str) -> int:
        try:
            num = nx.shortest_path_length(G, source, target)
        except:
            return float("inf")
        return num


class UtilsDrug:

    def readdrug_dicts(self, path):
        all_drug = []
        with open(path) as f:
            for index, line in enumerate(f.readlines()):
                if index == 0:
                    continue
                line = line.replace('"', "").split(",")
                all_drug.append(line[0])
        return all_drug

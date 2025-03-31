#!/usr/bin/env python

import argparse
import json
import pandas as pd
import numpy as np
from graph_utils import GraphUtils
from layer_utils import LayerUtils

class Process:
    def __init__(self, args):
        self.args = args

    def run(self):
        all_list_gene = pd.read_csv(self.args.path1).iloc[:, 0].to_list()
        global_graph = json.load(open(self.args.json, "r"))
        Gs = GraphUtils.trans_dicts2graph(graph_dicts=global_graph)

        focus_genes_list = self._load_focus_genes()
        drug_dicts = self._load_drug_dicts(all_list_gene)
        json.dump(drug_dicts, open(f"{self.args.out}_drug.json", "w"))

        first_layer_gene = drug_dicts[self.args.drug1] + drug_dicts[self.args.drug2]
        first_layer_gene.sort()

        removed_s, removed_t, transfer_layer = LayerUtils.trans_layer(
            Gs, first_layer_gene, focus_genes_list
        )

        second_layer_graph = GraphUtils.select_subgraph(global_graph, focus_genes_list)
        drug_list = self._filter_drug_list(drug_dicts, removed_s, global_graph)

        basic_layer_graph = GraphUtils.select_subgraph(global_graph, drug_list)
        removed = LayerUtils.max_connect(basic_layer_graph, second_layer_graph, transfer_layer)

        self._save_results(basic_layer_graph, second_layer_graph, transfer_layer, removed)

    def _load_focus_genes(self):
        focus_genes_list = []
        with open(self.args.path2) as f:
            for index, line in enumerate(f.readlines()):
                if index == 0:
                    continue
                focus_genes_list.append(line.replace('"', "").split()[0])
        return focus_genes_list

    def _load_drug_dicts(self, all_list_gene):
        drug_dicts = {}
        with open(self.args.path) as f:
            for index, line in enumerate(f.readlines()):
                if index == 0:
                    continue
                line = line.replace('"', "").split(",")
                drug_dicts[line[0]] = list(
                    set([m for m in line[1].strip().split() if m in all_list_gene])
                )
        return drug_dicts

    def _filter_drug_list(self, drug_dicts, removed_s, global_graph):
        drug_list = drug_dicts[self.args.drug1] + drug_dicts[self.args.drug2]
        drug_list.sort()
        degree_list = GraphUtils.count_degree(global_graph, drug_list)
        return [d for idx, d in enumerate(drug_list) if not (d in removed_s and degree_list[idx] == 0)]

    def _save_results(self, basic_layer_graph, second_layer_graph, transfer_layer, removed):
        transfer_layer = np.array(transfer_layer).T
        basic_layer_adj = np.array(basic_layer_graph['edges']).T
        second_layer_adj = np.array(second_layer_graph['edges']).T

        np.savez(f"{self.args.out}_transfer.npz", trans_layer=transfer_layer,
                 basic_layer_adj=basic_layer_adj, second_layer_adj=second_layer_adj)
        json.dump(basic_layer_graph, open(f"{self.args.out}_basic.json", "w"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/gene/merged_common_names.csv")
    parser.add_argument("--path1", type=str, default="data/gene/Combined_Datasets_Matrix.csv")
    parser.add_argument("--path2", type=str, default="data/gene/ML_gene.csv")
    parser.add_argument("--json", type=str, default="res.json")
    parser.add_argument("--drug1", type=str, default="mitoxantrone")
    parser.add_argument("--drug2", type=str, default="gambogic acid")
    parser.add_argument("--out", type=str, default="output")
    args = parser.parse_args()

    process = Process(args)
    process.run()





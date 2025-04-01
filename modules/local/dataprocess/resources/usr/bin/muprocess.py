#!/usr/bin/env python

import argparse
import json
import pandas as pd
import numpy as np
from graph_utils import GraphUtils
from layer_utils import LayerUtils, UtilsDrug
from itertools import combinations
import os
import logging
import concurrent.futures
from typing import List, Tuple


class MuProcess:
    def __init__(self, args, drug1, drug2, out):
        self.args = args
        self.drug1 = drug1
        self.drug2 = drug2
        self.out = out

    def _make_dir(self,path):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def process_drug_combination(args: argparse.Namespace, drug1: str, drug2: str, out: str) -> None:
        """处理单个药物组合的静态方法"""
        process = MuProcess(args, drug1, drug2, out)
        try:
            process.run()
        except Exception as e:
            logging.error(f"处理药物组合 {drug1}-{drug2} 时发生错误: {str(e)}")



    def run(self):
        self._make_dir(self.out)
        all_list_gene = pd.read_csv(self.args.path1).iloc[:, 0].to_list()
        global_graph = json.load(open(self.args.json, "r"))
        Gs = GraphUtils.trans_dicts2graph(graph_dicts=global_graph)

        focus_genes_list = self._load_focus_genes()
        drug_dicts = self._load_drug_dicts(all_list_gene)
        json.dump(drug_dicts, open(f"{self.out}/drug.json", "w"))

        first_layer_gene = drug_dicts[self.drug1] + drug_dicts[self.drug2]
        first_layer_gene.sort()

        removed_s, removed_t, transfer_layer = LayerUtils.trans_layer(
            Gs, first_layer_gene, focus_genes_list
        )


        second_layer_graph = GraphUtils.select_subgraph(global_graph, focus_genes_list)
        drug_list = self._filter_drug_list(drug_dicts, removed_s, global_graph)

        basic_layer_graph = GraphUtils.select_subgraph(global_graph, drug_list)
        first_layer_gene = basic_layer_graph["nodes_name"]
        removed = LayerUtils.max_connect(
            basic_layer_graph, second_layer_graph, transfer_layer
        )
        removed_s, removed_t, transfer_layer = LayerUtils.trans_layer(
            Gs, first_layer_gene, focus_genes_list
        )
        removed_all = removed_s + removed
        drug_list = [d for idx, d in enumerate(drug_list) if not d in removed_all]
        basic_layer_graph = GraphUtils.select_subgraph(global_graph, drug_list)
        first_layer_gene = basic_layer_graph["nodes_name"]
        removed_s, removed_t, transfer_layer = LayerUtils.trans_layer(
            Gs, first_layer_gene, focus_genes_list
        )
        self._save_results(
            basic_layer_graph, second_layer_graph, transfer_layer, removed
        )

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
        drug_list = drug_dicts[self.drug1] + drug_dicts[self.drug2]
        drug_list.sort()
        degree_list = GraphUtils.count_degree(global_graph, drug_list)
        return [
            d
            for idx, d in enumerate(drug_list)
            if not (d in removed_s and degree_list[idx] == 0)
        ]

    def _save_results(
        self, basic_layer_graph, second_layer_graph, transfer_layer, removed
    ):
        transfer_layer = np.array(transfer_layer).T
        basic_layer_adj = np.array(basic_layer_graph["edges"]).T
        second_layer_adj = np.array(second_layer_graph["edges"]).T

        np.savez(
            f"{self.out}/transfer.npz",
            trans_layer=transfer_layer,
            basic_layer_adj=basic_layer_adj,
            second_layer_adj=second_layer_adj,
        )
        json.dump(basic_layer_graph, open(f"{self.out}/basic.json", "w"))


def process_combinations_parallel(drug_combinations: List[Tuple[str, str]], args: argparse.Namespace, max_workers: int = None) -> None:
    """并行处理所有药物组合"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for drug1, drug2 in drug_combinations:
            out = f"./drug/{drug1.replace(' ','')}_{drug2.replace(' ','')}"

            future = executor.submit(
                MuProcess.process_drug_combination,
                args,
                drug1,
                drug2,
                out
            )
            futures.append(future)

        # 等待所有任务完成并处理异常
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"任务执行失败: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/gene/merged_common_names.csv")
    parser.add_argument("--path1", type=str, default="data/gene/Combined_Datasets_Matrix.csv")
    parser.add_argument("--path2", type=str, default="data/gene/ML_gene.csv")
    parser.add_argument("--json", type=str, default="res.json")
    parser.add_argument("--threads", type=int, default=50, help="并行处理的线程数，默认为CPU核心数")
    args = parser.parse_args()

    all_drug = UtilsDrug().readdrug_dicts(args.path)
    drug_combinations = list(combinations(all_drug, 2))

    # 使用多线程处理药物组合
    process_combinations_parallel(drug_combinations, args, max_workers=args.threads)


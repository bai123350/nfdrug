#!/usr/bin/env python

import argparse
import json
import csv
import numpy as np
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset
import os
import logging
from net import *
from sklearn.model_selection import train_test_split
import concurrent.futures
from typing import Tuple, Dict
import threading
from queue import Queue

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mumodel")


class DataSET(object):
    def __init__(self, base):
        self.basic_layer_graph = base
        self.csv_reader = None
        self.data_lock = threading.Lock()

    def _load_csv(self, file_path: str) -> None:
        with open(file_path) as f:
            self.csv_reader = list(csv.reader(f))

    def process_x(self) -> Tuple[np.ndarray, list]:
        if not self.csv_reader:
            self._load_csv(args.path1)

        with self.data_lock:
            first_line = True
            X_index = {}
            for line in self.csv_reader:
                if first_line:
                    temp_id_list = line[1:]
                    first_line = False
                else:
                    gene_name = line[0].strip()
                    if gene_name in self.basic_layer_graph["nodes_name"]:
                        X_index[gene_name] = np.array(line[1:], dtype=float)

            X_all = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(lambda g: X_index.get(g), gene_name)
                    for gene_name in self.basic_layer_graph["nodes_name"]
                ]
                X_all = [f.result() for f in futures if f.result() is not None]

            X_all = np.array(X_all).transpose()
            return X_all, temp_id_list


class GroupSet(object):
    def __init__(self, type):
        self.group = csv.reader(open(args.group))
        self.type = [row for row in list(type) if row != "Control"][0]

    def groupsplit(self):
        first_line = True
        y_tumor = {}  # 1-tumor 0-normal
        y_survival = {}

        p = 1000
        for line in self.group:
            if first_line:
                first_line = False
            else:
                sample_id = line[0].strip("\n").strip(" ")
                if line[1] == self.type:
                    y_tumor[sample_id] = 1
                elif line[1] == "Control":
                    y_tumor[sample_id] = 0
                if line[1] == self.type:
                    y_survival[sample_id] = (True, p)
                elif line[1] == "Control":
                    y_survival[sample_id] = (False, p)
                p += 1
        return y_tumor, y_survival


class AllData(object):
    def pall_data(self, x_sample_id_list, X_all, y_survival):
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
        y_data_label = np.expand_dims(np.array(y_data_label), axis=0)
        y_data_time = np.expand_dims(np.array(y_data_time), axis=0)
        y_data = np.concatenate((y_data_label, y_data_time), axis=0).T
        return x_data, y_data_label, y_data


class Utils:
    @staticmethod
    def normalize_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            scaler = preprocessing.StandardScaler()
            X_transformed = scaler.fit_transform(X)
            return X_transformed, y
        except Exception as e:
            logger.error(f"数据标准化失败: {str(e)}")
            raise


def build_bfregNN_model(
    gene_num: int,
    gene_num2: int,
    gene_adj: np.ndarray,
    gene_adj2: np.ndarray,
    transfer_layer: np.ndarray,
    device: torch.device
) -> BFRegNN_COX:
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            v1_future = executor.submit(lambda: torch.ones(gene_adj.shape[1], device=device))
            v2_future = executor.submit(lambda: torch.ones(gene_adj2.shape[1], device=device))
            v3_future = executor.submit(lambda: torch.ones(transfer_layer.shape[1], device=device))

            v1, v2, v3 = v1_future.result(), v2_future.result(), v3_future.result()

            ori_gene = torch.sparse_coo_tensor(gene_adj, v1, size=(gene_num, gene_num))
            ori_gene2 = torch.sparse_coo_tensor(gene_adj2, v2, size=(gene_num2, gene_num2))
            transfer_matrix = torch.sparse_coo_tensor(
                transfer_layer, v3, size=(gene_num, gene_num2)
            ).to_dense()

        model = BFRegNN_COX(
            gene_num, gene_num2, 64, ori_gene, transfer_matrix, ori_gene2
        ).to(device)
        return model
    except Exception as e:
        logger.error(f"模型构建失败: {str(e)}")
        raise


def count_gene():
    focus_genes_list = []
    with open(args.gene) as f:
        for index, line in enumerate(f.readlines()):
            if index == 0:
                continue
            focus_genes_list.append(line.replace('"', "").split()[0])
    return len(focus_genes_list)


def calculate_group():
    unique_values = set()
    with open(f"{args.group}", "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            unique_values.add(row[1])  # Assuming the second colum
    return unique_values


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def process_folder(p: str, batch_size: int) -> None:
    try:
        logger.info(f"处理文件夹: {p}")
        basic_layer_graph = json.load(open(os.path.join(p, "basic.json"), "r"))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            dataset = DataSET(basic_layer_graph)
            X_all_future = executor.submit(dataset.process_x)
            X_all, x_sample_id_list = X_all_future.result()

            unique_values = calculate_group()
            group_set = GroupSet(unique_values)
            y_future = executor.submit(group_set.groupsplit)
            y_tumor, y_survival = y_future.result()

            all_data = AllData()
            data_future = executor.submit(
                all_data.pall_data, x_sample_id_list, X_all, y_survival
            )
            x_data, y_data_label, y_data = data_future.result()

        make_dir(os.path.join("models", p))
        # 保存处理后的数据
        np.savez(
            f"{os.path.join('models', p, 'all.npz')}",
            x_data=x_data,
            y_data_label=y_data_label,
            y_data=y_data,
        )

        X, y = Utils().normalize_data(x_data, y_data)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 使用多线程创建数据加载器
        with concurrent.futures.ThreadPoolExecutor() as executor:
            train_dataset_future = executor.submit(GeneDataset, X_train, y_train)
            test_dataset_future = executor.submit(GeneDataset, X_test, y_test)

            train_data_dataset = train_dataset_future.result()
            test_data_dataset = test_dataset_future.result()

            train_data = DataLoader(train_data_dataset, batch_size=batch_size, shuffle=True)
            test_data = DataLoader(test_data_dataset, batch_size=batch_size, shuffle=False)

        # 保存数据和模型
        torch.save(train_data, os.path.join("models", p, "train_data.pt"))
        torch.save(test_data, os.path.join("models", p, "test_data.pt"))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        npz = np.load(f"{os.path.join(p, 'transfer.npz')}")
        model = build_bfregNN_model(
            X.shape[1],
            count_gene(),
            npz["basic_layer_adj"],
            npz["second_layer_adj"],
            npz["trans_layer"],
            device
        )
        torch.save(model, os.path.join("models", p, "model.pt"))

    except Exception as e:
        logger.error(f"处理文件夹 {p} 时发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path1", type=str, default="mart_export.txt")
    parser.add_argument("--folder", type=str, default="basic_layer_graph.json")
    parser.add_argument("--gene", type=str, default="len(gene)")
    parser.add_argument("--group", type=str, default="group.txt")
    parser.add_argument("--batch", type=int, default=10)
    args = parser.parse_args()

    # 使用线程池处理多个文件夹
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_folder, p, args.batch)
            for p in args.folder.split(",")
        ]
        concurrent.futures.wait(futures)

#!/usr/bin/env python

import argparse
import json
import csv
import numpy as np
from sklearn import preprocessing
from torch.utils.data import DataLoader
import os
import logging
from net import *
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mumodel")


class DataSET(object):
    def __init__(self, base):
        self.basic_layer_graph = base
        self.csv_reader = csv.reader(open(args.path1))

    def process_x(self):
        first_line = True
        X_index = {}
        for line in self.csv_reader:
            if first_line:
                temp_id_list = line[1:]
                first_line = False
            else:
                gene_name = line[0].strip("\n").strip(" ")
                if gene_name in self.basic_layer_graph["nodes_name"]:
                    X_index[gene_name] = line[1:]
        X_all = []
        for gene_name in self.basic_layer_graph["nodes_name"]:
            if gene_name not in X_index:
                continue
            X_all.append(X_index[gene_name])
        X_all = np.array(X_all)
        X_all = X_all.transpose()

        x_sample_id_list = []
        for sample_id in temp_id_list:
            x_sample_id_list.append(sample_id)
        return X_all, x_sample_id_list


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


class Utils(object):
    @staticmethod
    def normalize_data(X, y):
        scaler = preprocessing.StandardScaler().fit(X)
        X_transformed = scaler.transform(X)
        return X_transformed, y


def build_bfregNN_model(
    gene_num, gene_num2, gene_adj, gene_adj2, transfer_layer, device
):
    v1 = torch.ones(gene_adj.shape[1], device=device)
    ori_gene = torch.sparse_coo_tensor(gene_adj, v1, size=(gene_num, gene_num))

    v2 = torch.ones(gene_adj2.shape[1], device=device)
    ori_gene2 = torch.sparse_coo_tensor(gene_adj2, v2, size=(gene_num2, gene_num2))

    v3 = torch.ones(transfer_layer.shape[1], device=device)
    transfer_layer = torch.sparse_coo_tensor(
        transfer_layer, v3, size=(gene_num, gene_num2)
    ).to_dense()

    model = BFRegNN_COX(
        gene_num, gene_num2, 64, ori_gene, transfer_layer, ori_gene2
    ).to(device)
    return model


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path1", type=str, default="mart_export.txt")
    parser.add_argument("--folder", type=str, default="basic_layer_graph.json")
    parser.add_argument("--gene", type=str, default="len(gene)")
    parser.add_argument("--group", type=str, default="group.txt")
    parser.add_argument("--batch", type=int, default=10)
    args = parser.parse_args()
    for p in args.folder.split(","):
        logger.info(f"Loading {p}")
        basic_layer_graph = json.load(open(os.path.join(p, "basic.json"), "r"))
        X_all, x_sample_id_list = DataSET(basic_layer_graph).process_x()
        unique_values = calculate_group()
        y_tumor, y_survival = GroupSet(unique_values).groupsplit()
        x_data, y_data_label, y_data = AllData().pall_data(
            x_sample_id_list, X_all, y_survival
        )

        make_dir(os.path.join("models", p))
        np.savez(
            f"{os.path.join('models', p, 'all.npz')}",
            x_data=x_data,
            y_data_label=y_data_label,
            y_data=y_data,
        )
        X, y = Utils().normalize_data(x_data, y_data)
        # Split the data into training and testing sets

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Create DataLoader for training and testing sets
        train_data_dataset = GeneDataset(X_train, y_train)
        test_data_dataset = GeneDataset(X_test, y_test)

        train_data = DataLoader(train_data_dataset, batch_size=args.batch, shuffle=True)
        test_data = DataLoader(test_data_dataset, batch_size=args.batch, shuffle=False)
        # Save train_data and test_data
        torch.save(train_data, os.path.join("models", p, "train_data.pt"))
        torch.save(test_data, os.path.join("models", p, "test_data.pt"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        npz = np.load(f"{os.path.join(p, 'transfer.npz')}")
        model = build_bfregNN_model(X.shape[1], count_gene(),
                npz["basic_layer_adj"], npz["second_layer_adj"], npz["trans_layer"],
                device)
        torch.save(model, os.path.join("models", p, "model.pt"))

#!/usr/bin/env python

import argparse
import json
import csv
import numpy as np
from sklearn import preprocessing
from torch.utils.data import DataLoader
from net import *


class DataSET(object):
    def __init__(self):
        self.basic_layer_graph = json.load(open(args.path2, "r"))
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
    def __init__(self):
        self.group = csv.reader(open(args.group))

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
                if line[1] == args.disea:
                    y_tumor[sample_id] = 1
                elif line[1] == "Control":
                    y_tumor[sample_id] = 0
                if line[1] == args.disea:
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
    def normalize_data(X, y):
        scaler = preprocessing.StandardScaler().fit(X)
        X_transformed = scaler.transform(X)
        return X_transformed, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path1", type=str, default="mart_export.txt")
    parser.add_argument("--path2", type=str, default="basic_layer_graph.json")
    parser.add_argument("--group", type=str, default="group.txt")
    parser.add_argument("--disea", type=str, default="DKD")
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument('--out', type=str, default='xy')
    args = parser.parse_args()

    X_all, x_sample_id_list = DataSET().process_x()
    y_tumor, y_survival = GroupSet().groupsplit()

    x_data, y_data_label, y_data = AllData().pall_data(x_sample_id_list, X_all,y_survival)
    np.savez(f"{args.out}_all.npz", x_data = x_data, y_data_label = y_data_label, y_data = y_data)
    
    X, y = Utils().normalize_data(x_data, y_data)

    train_data_dataset = GeneDataset(X, y)
    train_data = DataLoader(train_data_dataset, batch_size=args.batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    


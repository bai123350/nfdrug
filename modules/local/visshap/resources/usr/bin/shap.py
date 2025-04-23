#!/usr/bin/env python

import argparse
import json
from math import log

# from pyexpat import model
from sympy import im
import torch
import torch.nn as nn
import logging
import os
import numpy as np
from net import *
import pandas as pd
from sanke import sankey


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Process:
    def __init__(self, args):
        self.file_list = args.dir.split(",")
        self.model_list = args.model.split(",")

    def evaluate_model(self, model, test_loader, device):
        model.eval()
        cross = nn.CrossEntropyLoss()
        test_loss = 0
        total_correct = 0
        total_samples = 0
        predictions = []
        true_labels = []

        x = np.array([])
        index_all = np.array([])

        with torch.no_grad():
            for d in test_loader:
                data = d[0].float().to(device)
                x = (
                    np.concatenate((x, data.cpu().numpy()), axis=0)
                    if x.size
                    else data.cpu().numpy()
                )
                label = d[1].float().to(device)
                event_label = label[:, 0]
                time_label = label[:, 1]

                pred1, pred, index = model(data, event_label, time_label)

                index_np = index.cpu().numpy()
                index_np = (
                    np.expand_dims(index_np, axis=0) if index_np.ndim == 0 else index_np
                )
                index_all = (
                    np.concatenate((index_all, index_np), axis=0)
                    if index_all.size
                    else index_np
                )
                loss = cross(pred, event_label)
                test_loss += loss.item()
                pred_labels = (pred >= 0.5).float()
                correct = (pred_labels == event_label).sum().item()
                total_correct += correct
                total_samples += event_label.size(0)

                predictions.extend(pred_labels.cpu().numpy())
                true_labels.extend(event_label.cpu().numpy())

        avg_loss = test_loss / len(test_loader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy, predictions, true_labels, x, index_all

    def top10(self):
        all_test_acc = {}
        for path in self.file_list:
            with open(os.path.join(path, "train_res.json"), "r") as f:
                data = json.load(f)
                all_test_acc[path.split("/")[1]] = data["test_accuracy"]
        all_test_acc = dict(
            sorted(all_test_acc.items(), key=lambda item: item[1], reverse=True)
        )
        top_10_acc = {}
        for index, (key, value) in enumerate(all_test_acc.items()):
            if index == 0 and int(value) != 1:
                top_10_acc = dict(list(all_test_acc.items())[:10])
                break
            if int(value) == 1:
                continue
            else:
                top_10_acc = dict(list(all_test_acc.items())[: (index + 1)])
                break
        return top_10_acc

    def get(self, p):
        for path in self.file_list:
            if path.split("/")[-1] == p:
                # train_data = torch.load(os.path.join(path, 'train_data.pt'), map_location='cpu')
                test_data = torch.load(os.path.join(path, "test_data.pt"))
                model = torch.load(os.path.join("modelfolder", p, "model.pt"))
                # Replace 'Net' with the actual model class used during training
                model.load_state_dict(torch.load(os.path.join(path, "best_model.pt")))
                test_loader = torch.utils.data.DataLoader(
                    test_data.dataset,
                    batch_size=min(test_data.batch_size, 16),
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True,
                )

                model = model.to("cuda:0")
                test_loss, test_accuracy, predictions, true_labels, x, index_all = (
                    self.evaluate_model(model, test_loader, "cuda:0")
                )

                all_res = {}
                for pp in range(x.shape[0]):
                    sample_input = torch.FloatTensor(x[pp : (pp + 1)]).to("cuda:0")
                    target_class = index_all[pp]
                    # 计算梯度
                    ig = integrated_gradients(
                        model, sample_input, target_class=target_class, steps=50
                    )
                    importance_scores = np.abs(ig).sum(axis=0)
                    top_5_indices = np.argsort(importance_scores)[-5:][::-1]
                    top_5_dict = dict(
                        zip(top_5_indices, importance_scores[top_5_indices])
                    )

                    all_res[pp] = top_5_dict

        return x, index_all, all_res


class AllDir(object):
    def __init__(self, args):
        self.file_list = args.all.split(",")

    def get(self, p):
        for path in self.file_list:
            if p == path.split("/")[1]:
                basic = json.load(open(os.path.join(path, "basic.json"), "r"))
                drug = json.load(open(os.path.join(path, "drug.json"), "r"))
                drug = {k.replace(" ", ""): v for k, v in drug.items()}
                logger.info(
                    f"{p.split('_')[0]}-{drug[p.split('_')[0]]},{len(drug[p.split('_')[0]])}"
                )
                logger.info(
                    f"{p.split('_')[1]}-{drug[p.split('_')[1]]},{len(drug[p.split('_')[1]])}"
                )
                # all_test_acc.update({"basic": basic, "drug": drug})
                input_data = basic["nodes_name"]

        return input_data


def integrated_gradients(
    model, input_tensor, target_class, event_label=None, time_label=None, steps=50
):
    """
    计算给定输入和目标类的积分梯度。

    Args:
        model: 神经网络模型
        input_tensor: 输入张量 [batch_size, feature_dim]
        target_class: 目标类别索引
        event_label: 事件标签 (可选)
        time_label: 时间标签 (可选)
        steps: 近似步数
    Returns:
        integrated_grads: 输入的积分梯度
    """
    model.eval()
    baseline = torch.zeros_like(input_tensor).to(input_tensor.device)
    scaled_inputs = [
        baseline + (float(i) / steps) * (input_tensor - baseline)
        for i in range(steps + 1)
    ]
    grads = []

    for scaled_input in scaled_inputs:
        scaled_input.requires_grad = True
        if event_label is None:
            event_label = torch.zeros(scaled_input.shape[0]).to(scaled_input.device)
        if time_label is None:
            time_label = torch.zeros(scaled_input.shape[0]).to(scaled_input.device)

        output, out1, _ = model(scaled_input, event_label, time_label)
        if output.dim() == 0:
            output = output.unsqueeze(0)

        score = output[target_class]
        gradients = torch.autograd.grad(
            score, scaled_input, create_graph=True, retain_graph=True
        )[0]
        grads.append(gradients.cpu().detach().numpy())

    # 计算平均梯度
    avg_grads = np.average(grads[:-1], axis=0)
    integrated_grads = (
        input_tensor.cpu().detach().numpy() - baseline.cpu().detach().numpy()
    ) * avg_grads
    return integrated_grads


def read_gene(path):
    focus_genes_list = []
    with open(path) as f:
        for index, line in enumerate(f.readlines()):
            if index == 0:
                continue
            focus_genes_list.append(line.replace('"', "").split()[0])
    return focus_genes_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="")
    parser.add_argument("--all", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--gene", type=str, default="")
    args = parser.parse_args()

    data_util = Process(args)
    results = data_util.top10()
    x_test, index_all, all_res = data_util.get(list(results.keys())[0])
    all_dir = AllDir(args)
    input_data = all_dir.get(list(results.keys())[0])
    biao_gene = read_gene(args.gene)
    ss = np.array([])
    for k, v in all_res.items():
        gene = biao_gene[index_all[int(k)]]
        for k1, v1 in v.items():
            ss = (
                np.vstack((ss, [input_data[int(k1)], gene, v1]))
                if ss.size
                else np.array([[input_data[int(k1)], gene, v1]])
            )
    df = pd.DataFrame(ss, columns=["input_data", "biao_gene", "value"])
    df = df.loc[df.groupby(['input_data', 'biao_gene'])['value'].idxmax()]
    sankey(
        left=df["input_data"],
        right=df["biao_gene"],
        rightWeight=df["value"].astype(float),
        aspect=20,
        fontsize=3,
        figureName=f"{list(results.keys())[0]}",
    )
    df.to_csv(f"{list(results.keys())[0]}.csv", index=False)

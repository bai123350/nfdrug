#!/usr/bin/env python

import argparse
import json
from pyexpat import model
from sympy import im
import torch
import torch.nn as nn
import logging
import os
from net import *


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Process:
    def __init__(self, args):
        self.file_list = args.dir.split(",")

    def evaluate_model(self ,model, test_loader, device):
        # model.eval()
        cross = nn.CrossEntropyLoss()
        test_loss = 0
        total_correct = 0
        total_samples = 0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for d in test_loader:
                data = d[0].float().to(device)
                label = d[1].float().to(device)
                event_label = label[:, 0]
                time_label = label[:, 1]

                pred = model(data, event_label, time_label)
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
        return avg_loss, accuracy, predictions, true_labels

    def top10(self):
        all_test_acc = {}
        for path in self.file_list:
            with open(os.path.join(path ,'train_res.json'), 'r') as f:
                data = json.load(f)
                all_test_acc[path.split("/")[1]] = data['test_accuracy']
        all_test_acc = dict(sorted(all_test_acc.items(), key=lambda item: item[1], reverse=True))
        top_10_acc = {}
        for index, (key, value) in enumerate(all_test_acc.items()):
            if index == 0 and int(value) != 1:
                top_10_acc = dict(list(all_test_acc.items())[:10])
                break
            if int(value) == 1: continue
            else:
                top_10_acc = dict(list(all_test_acc.items())[:(index + 1)])
                break
        return top_10_acc

    def get(self, p):
        for path in self.file_list:
            if path.split('/')[-1] == p:
                # train_data = torch.load(os.path.join(path, 'train_data.pt'), map_location='cpu')
                test_data = torch.load(os.path.join(path, 'test_data.pt'))
                # test_data = test_data.to('cuda:0')
                test_loader = torch.utils.data.DataLoader(
                        test_data.dataset,
                        batch_size=min(test_data.batch_size, 16),
                        shuffle=False,
                        num_workers=2,
                        pin_memory=True
                    )
                model = torch.load(os.path.join(path, 'best_model.pt'))
                # model = model.to('cuda:0')
                test_loss, test_accuracy, predictions, true_labels = self.evaluate_model(model, test_loader, "cuda:0")



class AllDir(object):
    def __init__(self, args):
        self.file_list = args.all.split(",")

    def get(self,p):
        all_test_acc = {}
        for path in self.file_list:
            if p == path.split("/")[1]:
                basic = json.load(open(os.path.join(path, 'basic.json'), 'r'))
                drug = json.load(open(os.path.join(path, 'drug.json'), 'r'))
                drug = {k.replace(' ',''): v for k, v in drug.items()}
                logger.info(f"{drug}")
                # all_test_acc.update({"basic": basic, "drug": drug})

        return all_test_acc

def compute_gradients(model, inputs, target_class):
    """
    Compute gradients of the output with respect to the inputs for a specific target class.
    """
    model.eval()
    inputs.requires_grad = True

    outputs = model(inputs)
    loss = outputs[0, target_class]
    loss.backward()

    gradients = inputs.grad
    return gradients





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="")
    parser.add_argument('--all', type=str, default="")
    args = parser.parse_args()

    data_util = Process(args)
    results = data_util.top10()
    data_util.get(list(results.keys())[0])
    # logger.info(f"Top 10 results: {results}")
    all_dir = AllDir(args)
    all_dir.get(list(results.keys())[0])


    tttt







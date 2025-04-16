#!/usr/bin/env python

import argparse
import json
from sympy import im
import torch
import torch.nn as nn
import logging
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Process:
    def __init__(self, args):
        self.file_list = args.dir.split(",")

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
    logger.info(f"Top 10 results: {results}")


    tttt





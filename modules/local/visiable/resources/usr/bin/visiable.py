#!/usr/bin/env python

import argparse
import json
import logging
import os
import re

import torch

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
                all_test_acc[path] = data['test_accuracy']
        all_test_acc = dict(sorted(all_test_acc.items(), key=lambda item: item[1], reverse=True))
        top_10_acc = dict(list(all_test_acc.items())[:10])
        return top_10_acc

class DataUtil(Process):
    def read_data(self):
        for p in self.file_list:
            train_data = torch.load(os.path.join(p, 'train_data.pt'), map_location='cpu')
            test_data = torch.load(os.path.join(p, 'test_data.pt'), map_location='cpu')

        return ""



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="")
    args = parser.parse_args()
    top10 = Process(args).top10()








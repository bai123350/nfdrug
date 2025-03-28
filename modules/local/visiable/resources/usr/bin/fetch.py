#!/usr/bin/env python

import argparse
import json




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--score', type=float, default=0.5)
    parser.add_argument('--path1', type=str, default='/home/bio-17/projects/drug/nf_drug/nfdrug/data/9606.protein.info.v11.5.txt')
    parser.add_argument('--path2', type=str, default='/home/bio-17/projects/drug/nf_drug/nfdrug/data/9606.protein.links.full.v11.5.txt')
    parser.add_argument('--out', type=str, default='graph_dicts.json')
    args = parser.parse_args()





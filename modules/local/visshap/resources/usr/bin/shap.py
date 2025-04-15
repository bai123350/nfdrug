#!/usr/bin/env python

import argparse
import json


def read_global_graph(thres_score,path1,path2):
    protein_id_dicts = {}

    with open(path1) as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                continue
            line = line.split()
            protein_id_dicts[line[0]] = line[1]
    
    protein_graphs = {}
    protein_graphs['nodes_name'] = []
    protein_graphs['nodes'] = []
    protein_graphs['edges'] = []
    protein_graphs['edge_att'] = []

    graph_dicts = {}
    graph_dicts['nodes'] = {}
    with open(path2) as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                continue
            line = line.split()
            source = protein_id_dicts[line[0]]
            target = protein_id_dicts[line[1]]
            score = float(line[-1])/1000
            if score >= thres_score:
                if source not in protein_graphs['nodes_name']:
                    protein_graphs['nodes_name'].append(source)
                    protein_graphs['nodes'].append(protein_graphs['nodes_name'].index(source))

                if target not in protein_graphs['nodes_name']:
                    protein_graphs['nodes_name'].append(target)
                    protein_graphs['nodes'].append(protein_graphs['nodes_name'].index(target))
                if source not in graph_dicts:
                    graph_dicts[source] = []
                graph_dicts[source].append(target)
                protein_graphs['edges'].append([source,target])
                protein_graphs['edge_att'].append(score)
    return graph_dicts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--score', type=float, default=0.5)
    parser.add_argument('--path1', type=str, default='/home/bio-17/projects/drug/nf_drug/nfdrug/data/9606.protein.info.v11.5.txt')
    parser.add_argument('--path2', type=str, default='/home/bio-17/projects/drug/nf_drug/nfdrug/data/9606.protein.links.full.v11.5.txt')
    parser.add_argument('--out', type=str, default='graph_dicts.json')
    args = parser.parse_args()
    graph_dicts = read_global_graph(args.score,args.path1,args.path2)
    json.dump(graph_dicts, open(args.out,'w'))





#!/usr/bin/env python3
# coding: utf-8
import random
import torch
import numpy as np
from scipy.spatial import distance
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

num_nodes = 10000
# Number of labels

L = 10
# Average number of labels of each node
k = 3
# homophily parameter
alpha = 0.0
#
b = 1

# initialization
labels = torch.zeros([num_nodes, L], dtype=torch.int32)
# assign k label to each node randomly
for index, nodes in enumerate(labels):
    i = 0
    while i == 0:
        for label in range(L):
            prob = random.uniform(0, 1)
            if prob <= k / L:
                labels[index][label] = 1
                i += 1


# initialization
adj = torch.zeros([num_nodes, num_nodes], dtype=torch.int8)
# construct the graph adj: symmetric
for ind1 in range(0, num_nodes-1):
    for ind2 in range(ind1+1, num_nodes):
        # p(i,j)
        dist = distance.hamming(labels[ind1], labels[ind2])
        # p = 1 / (2 * (1 + pow(pow(b, -1) * dist, alpha)))
        p = 1 / (2 * (1+pow(pow(b, -1)*dist, alpha)))
        prob = random.uniform(0, 1)
        if prob <= p:
            adj[ind1][ind2] = 1
            adj[ind2][ind1] = 1
# print(adj)
# construct edge list
edge_list = torch.nonzero(adj)
print("graph has", edge_list.shape[0]/2, "edges")
edge_index = torch.stack((edge_list[:, 0], edge_list[:, 1]), dim=0)


labels_dict = {}
no_label_count = 0
for inde, label in enumerate(labels):
    labels_dict[inde] = set(np.array(torch.flatten(torch.nonzero(label))))
    if len(labels_dict[inde]) == 0:
        no_label_count += 1
print(no_label_count, "nodes have no labels in this graph")


# ouput the homophily ratio:
homo_count = 0
edges = pd.DataFrame(torch.nonzero(adj).tolist())
for index, edge in edges.iterrows():
    # self loops dont count
    if edge[0] == edge[1]:
        continue
    else:
        l1 = labels_dict[edge[0].item()]
        l2 = labels_dict[edge[1].item()]

        co_l = set(l1).intersection(set(l2))

        # if one node has no labels:
        if len(l1) == 0 or len(l2) == 0:
            continue
        elif (len(co_l) / len(l1) > 0.9) and (len(co_l) / len(l2) > 0.9):
            homo_count += 1

homo_ratio = homo_count / edge_list.shape[0]
print("alpha:{:2.1f} b:{:3.2f} homo-ratio:{:4.3f}".format(alpha, b, homo_ratio))




















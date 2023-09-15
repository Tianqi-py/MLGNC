
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
import os
from sklearn.metrics import jaccard_score


# Average number of labels of each node
#k = 4
# homophily parameter
alpha = 9.5
#
b = 0.05
path = 'D:/code/data/hyperspheres_10_10_0'

labels = np.genfromtxt(os.path.join(path,  "labels.csv"), skip_header=1,
                       dtype=np.dtype(float), delimiter=',')
num_nodes = labels.shape[0]


# initialization
adj = torch.zeros([num_nodes, num_nodes], dtype=torch.int8)
# construct the graph adj: symmetric
for ind1 in range(0, num_nodes-1):
    for ind2 in range(ind1+1, num_nodes):
        # p(i,j)
        dist = distance.hamming(labels[ind1], labels[ind2])
        # p = 1 / (2 * (1 + pow(pow(b, -1) * dist, alpha)))
        p = 1 / (30 * (1+pow(pow(b, -1)*dist, alpha)))
        prob = random.uniform(0, 1)
        if prob <= p:
            adj[ind1][ind2] = 1
            adj[ind2][ind1] = 1


# construct edge list
edge_list = torch.nonzero(adj)
print("graph has", edge_list.shape[0]/2, "edges")
edge_index = torch.stack((edge_list[:, 0], edge_list[:, 1]), dim=0)
edges = torch.transpose(edge_index, 0, 1).long()

support = 0.0
for i, edge in enumerate(edges):
    support = support + jaccard_score(labels[edge[0].item()],
                                      labels[edge[1].item()])
homo_ratio = support / edges.shape[0]

print("alpha:{:2.1f} b:{:3.2f} homo-ratio:{:4.3f}".format(alpha, b, homo_ratio))


torch.save(edge_index, os.path.join(path, "edge_index.pt"))

















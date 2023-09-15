import scipy.io
import torch
import numpy as np
from sklearn.metrics import jaccard_score
import torch
from torch_geometric.datasets import Yelp

# mat data
# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """Convert a scipy sparse matrix to a torch sparse tensor."""
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     indices = torch.from_numpy(
#         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse.FloatTensor(indices, values, shape)

# mat = scipy.io.loadmat('..//data//flickr.mat')
# labels = mat['group']
# adj = mat['network']
# adj = sparse_mx_to_torch_sparse_tensor(adj)
# labels = sparse_mx_to_torch_sparse_tensor(labels)
# edges = torch.nonzero(adj.to_dense())
# #homo ratio
# support = 0.0
# for i, edge in enumerate(edges):
#     support = support + jaccard_score(labels[edge[0].item()].to_dense(),
#                                       labels[edge[1].item()].to_dense())


print('Loading dataset ' + 'Yelp...')
dataset = Yelp(root='/tmp/Yelp')
data = dataset[0]
labels = data.y
edge_index = data.edge_index
edges = torch.transpose(edge_index, 0, 1)
#homo ratio
support = 0.0
for i, edge in enumerate(edges):
    support = support + jaccard_score(labels[edge[0].item()],
                                      labels[edge[1].item()])

h = support / edges.shape[0]
print(h)

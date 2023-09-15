import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Conv1d, MaxPool1d, Linear
import os
import torch
import scipy
import scipy.io
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.datasets import Yelp
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader
import argparse
from torch_geometric.utils import degree as Degree
import networkx as nx
from torch_geometric.utils import to_networkx
import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from earlystopping import EarlyStopping
from torch.utils.data import DataLoader
from metric.metrics import f1_loss, BCE_loss, _eval_rocauc, ap_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default='dblp',
                        help='Name of the dataset'
                        'Hyperspheres_64_64_0'
                        'pcg_removed_isolated_nodes'
                        'Humloc')
    parser.add_argument('--train_percent', type=float, default=0.6,
                        help='percentage of data used for training')
    parser.add_argument("--split_name", default='split_0.pt',
                        help='Name of the split')
    parser.add_argument("--model_name", default='LANC',
                        help='LFLF_GCN or LFLF_GAT or LFLF_SAGE')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument("--device_name", default='cuda',
                        help='Name of the device')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--layer', type=float, default=2,
                        help='number of layer in LFLF')
    parser.add_argument('--patience', type=float, default=100,
                        help='patience for early stopping.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='patience for early stopping.')
    return parser.parse_args()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_mat_data(data_name, split_name, train_percent, path="../../data/"):
    print('Loading dataset ' + data_name + '.mat...')
    mat = scipy.io.loadmat(path + data_name)
    labels = mat['group']
    labels = sparse_mx_to_torch_sparse_tensor(labels).to_dense()

    adj = mat['network']
    adj = sparse_mx_to_torch_sparse_tensor(adj).long()
    #edge_index = torch.transpose(torch.nonzero(adj.to_dense()), 0, 1).long()

    # prepare the feature matrix
    #features = torch.range(0, labels.shape[0] - 1).long()
    features = torch.randn(labels.shape[0], 128)

    #(10312, 64) --->(1, 10312,64)
    #features = features.view(1, -1, 64)

    folder_name = data_name + "_" + str(train_percent)
    file_path = os.path.join(path, folder_name, split_name)
    masks = torch.load(file_path)
    train_idx = masks["train_mask"]
    train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True

    val_idx = masks["val_mask"]
    val_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    val_mask[val_idx] = True

    test_idx = masks["test_mask"]
    test_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True

    #num_class = labels.shape[1]
    num_nodes = labels.shape[0]

    G = Data(x=features,
             edge_index=adj._indices(),
             y=labels)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    G.adj = adj
    # label embedding to start with
    lbl_emb = torch.arange(labels.shape[1]).long()
    G.lbl_emb = lbl_emb
    G.n_id = torch.arange(num_nodes)

    # maximum degree
    degree = 0
    adj_dense = adj.to_dense()
    for row in adj_dense:
        deg = torch.flatten(torch.nonzero(row)).shape[0]
        if deg > degree:
            degree = deg

    # prepare the feature matrix
    attrs = []
    for i in range(labels.shape[0]):
        neigh_ind = torch.flatten(torch.nonzero(adj_dense[i]))
        if len(neigh_ind) < degree:
            num_to_pad = degree - len(neigh_ind)
            padding = torch.zeros(num_to_pad, features.shape[1])
            #featues of neighbors
            attr = features[neigh_ind]
            attr = torch.vstack((attr, padding))
        else:
            attr = features[neigh_ind]
            # featues of neighbors
        attrs.append(attr)

    G.x = torch.stack(attrs)
    print(G.x.shape)
    return G


def load_pcg(data_name, split_name, train_percent, path="../../data/"):
    print('Loading dataset ' + data_name + '.csv...')

    labels = np.genfromtxt(os.path.join(path, data_name, "labels.csv"),
                           dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()
    features = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "features.csv"),
                            dtype=np.dtype(float), delimiter=',')).float()

    edges = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "edges_undir.csv"),
                                       dtype=np.dtype(float), delimiter=','))
    edge_index = torch.transpose(edges, 0, 1).long()
    adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]),
                                  (features.shape[0], features.shape[0]))

    folder_name = data_name + "_" + str(train_percent)
    file_path = os.path.join(path, folder_name, split_name)
    masks = torch.load(file_path)
    train_idx = masks["train_mask"]
    train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True

    val_idx = masks["val_mask"]
    val_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    val_mask[val_idx] = True

    test_idx = masks["test_mask"]
    test_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True

    #num_class = labels.shape[1]
    num_nodes = labels.shape[0]

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    G.adj = adj

    # label embedding to start with
    lbl_emb = torch.arange(labels.shape[1]).long()
    G.lbl_emb = lbl_emb
    G.n_id = torch.arange(num_nodes)

    # maximum degree
    degree = 0
    adj_dense = adj.to_dense()
    for row in adj_dense:
        deg = torch.flatten(torch.nonzero(row)).shape[0]
        if deg > degree:
            degree = deg

    # prepare the feature matrix
    attrs = []
    for i in range(labels.shape[0]):
        neigh_ind = torch.flatten(torch.nonzero(adj_dense[i]))
        if len(neigh_ind) < degree:
            num_to_pad = degree - len(neigh_ind)
            padding = torch.zeros(num_to_pad, features.shape[1])
            # featues of neighbors
            attr = features[neigh_ind]
            attr = torch.vstack((attr, padding))
        else:
            attr = features[neigh_ind]
            # featues of neighbors
        attrs.append(attr)

    G.x = torch.stack(attrs)
    print(G.x.shape)

    return G


def load_humloc(data_name="HumanGo", path="../../data/"):
    edge_list = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "edge_list.csv"),
                                           skip_header=1, dtype=np.dtype(float), delimiter=','))[:, :2].long()
    edge_list_other_half = torch.hstack((edge_list[:, 1].reshape(-1, 1), edge_list[:, 0].reshape(-1, 1)))
    edge_index = torch.transpose(edge_list, 0, 1)
    edge_index_other_half = torch.transpose(edge_list_other_half, 0, 1)
    edge_index = torch.hstack((edge_index, edge_index_other_half))

    labels = np.genfromtxt(os.path.join(path, data_name, "labels.csv"),
                           dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()

    features = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "features.csv"),
                            dtype=np.dtype(float), delimiter=',')).float()

    adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]),
                                  (features.shape[0], features.shape[0]))

    file_path = os.path.join(path, data_name, "split.pt")
    masks = torch.load(file_path)
    train_idx = masks["train_mask"]
    train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True

    val_idx = masks["val_mask"]
    val_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    val_mask[val_idx] = True

    test_idx = masks["test_mask"]
    test_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    G.adj = adj

    num_nodes = labels.shape[0]
    # label embedding to start with
    lbl_emb = torch.arange(labels.shape[1]).long()
    G.lbl_emb = lbl_emb
    G.n_id = torch.arange(num_nodes)

    # maximum degree
    degree = 0
    adj_dense = adj.to_dense()
    for row in adj_dense:
        deg = torch.flatten(torch.nonzero(row)).shape[0]
        if deg > degree:
            degree = deg

    # prepare the feature matrix
    attrs = []
    for i in range(labels.shape[0]):
        neigh_ind = torch.flatten(torch.nonzero(adj_dense[i]))
        if len(neigh_ind) < degree:
            num_to_pad = degree - len(neigh_ind)
            padding = torch.zeros(num_to_pad, features.shape[1])
            # featues of neighbors
            attr = features[neigh_ind]
            attr = torch.vstack((attr, padding))
        else:
            attr = features[neigh_ind]
            # featues of neighbors
        attrs.append(attr)

    G.x = torch.stack(attrs)
    print(G.x.shape)

    return G


def load_eukloc(data_name="EukaryoteGo", path="../../data/"):
    edge_list = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "edge_list.csv"),
                                           skip_header=1, dtype=np.dtype(float), delimiter=','))[:, :2].long()
    edge_list_other_half = torch.hstack((edge_list[:, 1].reshape(-1, 1), edge_list[:, 0].reshape(-1, 1)))
    edge_index = torch.transpose(edge_list, 0, 1)
    edge_index_other_half = torch.transpose(edge_list_other_half, 0, 1)
    edge_index = torch.hstack((edge_index, edge_index_other_half))

    labels = np.genfromtxt(os.path.join(path, data_name, "labels.csv"),
                           dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()

    adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]),
                                  (labels.shape[0], labels.shape[0]))

    features = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "features.csv"),
                            dtype=np.dtype(float), delimiter=',')).float()

    file_path = os.path.join(path, data_name, "split.pt")
    masks = torch.load(file_path)
    train_idx = masks["train_mask"]
    train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True

    val_idx = masks["val_mask"]
    val_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    val_mask[val_idx] = True

    test_idx = masks["test_mask"]
    test_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask

    G.adj = adj

    num_nodes = labels.shape[0]
    # label embedding to start with
    lbl_emb = torch.arange(labels.shape[1]).long()
    G.lbl_emb = lbl_emb
    G.n_id = torch.arange(num_nodes)

    # maximum degree
    degree = 0
    adj_dense = adj.to_dense()
    for row in adj_dense:
        deg = torch.flatten(torch.nonzero(row)).shape[0]
        if deg > degree:
            degree = deg

    # prepare the feature matrix
    attrs = []
    for i in range(labels.shape[0]):
        neigh_ind = torch.flatten(torch.nonzero(adj_dense[i]))
        if len(neigh_ind) < degree:
            num_to_pad = degree - len(neigh_ind)
            padding = torch.zeros(num_to_pad, features.shape[1])
            # featues of neighbors
            attr = features[neigh_ind]
            attr = torch.vstack((attr, padding))
        else:
            attr = features[neigh_ind]
            # featues of neighbors
        attrs.append(attr)

    G.x = torch.stack(attrs)
    print(G.x.shape)

    return G


def import_ogb(data_name):
    print('Loading dataset ' + data_name + '...')

    dataset = PygNodePropPredDataset(name=data_name, transform=T.ToSparseTensor(attr='edge_attr'))
    data = dataset[0]
    data.x = data.adj_t.mean(dim=1)
    data.adj_t.set_value_(None)

    row, col, _ = data.adj_t.coo()
    edge_index = torch.vstack((row, col))

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    train_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True

    val_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    val_mask[valid_idx] = True

    test_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True

    num_nodes = data.x.shape[0]

    G = Data(x=data.x,
             edge_index=edge_index,
             y=data.y)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    G.n_id = torch.arange(num_nodes)

    # label embedding to start with
    lbl_emb = torch.arange(G.y.shape[1]).long()
    G.lbl_emb = lbl_emb
    G.n_id = torch.arange(num_nodes)

    # maximum degree
    #calculate degree of the graph
    degrees = Degree(edge_index[0], G.num_nodes, dtype=torch.long)
    degree = torch.max(degrees).item()

    # prepare the feature matrix
    neighbor_loader = NeighborLoader(G,
                                     num_neighbors=[-1],
                                     batch_size=64,
                                     shuffle=False,
                                     input_nodes=G.n_id)
    # attrs = []
    # for batch in neighbor_loader:
    #
    # attrs = []
    # for i in range(G.y.shape[0]):
    #     neigh_ind = torch.flatten(torch.nonzero(adj_dense[i]))
    #     if len(neigh_ind) < degree:
    #         num_to_pad = degree - len(neigh_ind)
    #         padding = torch.zeros(num_to_pad, G.x.shape[1])
    #         # featues of neighbors
    #         attr = G.x[neigh_ind]
    #         attr = torch.vstack((attr, padding))
    #     else:
    #         attr = G.x[neigh_ind]
    #         # featues of neighbors
    #     attrs.append(attr)
    #
    # G.x = torch.stack(attrs)
    # print(G.x.shape)

    return G


def load_dblp(data_name, split_name, train_percent, path="../../../data/dblp/"):
    labels = np.genfromtxt("../../../data/dblp/labels.txt",
                           dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()
    features = torch.FloatTensor(np.genfromtxt(os.path.join(path, "features.txt"),
                                               delimiter=",", dtype=np.float64))
    edge_list = torch.tensor(np.genfromtxt(os.path.join(path, "dblp.edgelist"))).long()
    edge_list_other_half = torch.hstack((edge_list[:, 1].reshape(-1, 1), edge_list[:, 0].reshape(-1, 1)))
    edge_index = torch.transpose(edge_list, 0, 1)
    edge_index_other_half = torch.transpose(edge_list_other_half, 0, 1)
    edge_index = torch.hstack((edge_index, edge_index_other_half))

    adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]),
                                  (labels.shape[0], labels.shape[0]))

    #folder_name = path + data_name + "_" + str(train_percent)
    file_path = os.path.join("../../../data/dblp/dblp_0.6", split_name)
    masks = torch.load(file_path)
    train_idx = masks["train_mask"]
    train_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True

    val_idx = masks["val_mask"]
    val_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
    val_mask[val_idx] = True

    test_idx = masks["test_mask"]
    test_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask

    G.adj = adj

    num_nodes = labels.shape[0]
    # label embedding to start with
    lbl_emb = torch.arange(labels.shape[1]).long()
    G.lbl_emb = lbl_emb
    G.n_id = torch.arange(num_nodes)

    # maximum degree
    degree = 0
    adj_dense = adj.to_dense()
    for row in adj_dense:
        deg = torch.flatten(torch.nonzero(row)).shape[0]
        if deg > degree:
            degree = deg

    # prepare the feature matrix
    attrs = []
    for i in range(labels.shape[0]):
        neigh_ind = torch.flatten(torch.nonzero(adj_dense[i]))
        if len(neigh_ind) < degree:
            num_to_pad = degree - len(neigh_ind)
            padding = torch.zeros(num_to_pad, features.shape[1])
            # featues of neighbors
            attr = features[neigh_ind]
            attr = torch.vstack((attr, padding))
        else:
            attr = features[neigh_ind]
            # featues of neighbors
        attrs.append(attr)

    G.x = torch.stack(attrs)
    print(G.x.shape)

    return G


# class Attention(nn.Module):
#     def __init__(self, in_size, hidden_size=16):
#         super(Attention, self).__init__()
#
#         self.project = nn.Sequential(
#             nn.Linear(in_size, hidden_size),
#             nn.Tanh(),
#             nn.Linear(hidden_size, 1, bias=False)
#         )
#
#     def forward(self, z):
#         # unsoftmaxed attention vector
#         w = self.project(z)
#         # softmax attention vector
#         beta = torch.softmax(w, dim=1)
#         return (beta * z).sum(1), beta


class LANC(torch.nn.Module):
    def __init__(self, in_channels, class_channels, num_label):
        super().__init__()
        self.conv1 = Conv1d(in_channels, 16, 2)
        self.conv2 = Conv1d(in_channels, 16, 3)
        self.conv3 = Conv1d(in_channels, 16, 4)
        self.conv4 = Conv1d(in_channels, 16, 5)
        self.mlp = nn.Sequential(
                                Linear(192, 64),
                                nn.ReLU(),
                                Linear(64, class_channels))
        self.mlp1 = nn.Sequential(Linear(128, 64),
                                  nn.ReLU(),
                                  Linear(64, class_channels))

        self.attention = nn.Sequential(nn.Linear(192, 64),
                                       nn.Tanh(),
                                       nn.Linear(64, 1, bias=False))
        self.lbl_emb = nn.Embedding(num_label, 128)

    def forward(self, x, y):
        y = self.lbl_emb(y)

        x = x.permute(0, 2, 1)
        #convolution
        x1 = self.conv1(x)

        x1 = F.relu(x1)
        #print(x1[0])
        x1 = F.dropout(x1, p=0.6)
        #print(x1[0])
        #x1 = F.dropout(F.relu(self.conv1(x)), p=0.6)

        #print(x1[0])
        #print("%%%%%%%%%%%%%%%%%%%%%%")
        x2 = F.dropout(F.relu(self.conv2(x)), p=0.6)
        x3 = F.dropout(F.relu(self.conv3(x)), p=0.6)
        x4 = F.dropout(F.relu(self.conv4(x)), p=0.6)
        #print("%%%%%%%%%%%%%%%%%%%%")
        #print(x1.shape)
        #max pooling
        x1 = torch.amax(x1, 2)
        x2 = torch.amax(x2, 2)
        x3 = torch.amax(x3, 2)
        x4 = torch.amax(x4, 2)
        # feature vector
        out = torch.cat((x1, x2, x3, x4), dim=1)
        # (64, 64)
        #print(out[0:3])
        # attention vector
        # (batch_size, node embedding + label embedding dimension)
        s = []
        for i in range(y.shape[0]):
            # 64 is the batch size
            support = torch.cat(out.shape[0] * [y[i]]).reshape(out.shape[0], -1)
            # concat the node emb with one label emb
            c = torch.hstack((out, support))
            s.append(c)
        #print(torch.stack(s).shape)
        s = self.attention(torch.stack(s))#.squeeze()
        #print(s.shape)
        s = s.squeeze()
        #print('s')
        #print(s.shape)
        #print(s[0:3])
        a = F.softmax(torch.transpose(s, 0, 1), dim=1)
        #print("a")
        #print(a[0:3])

        att_vec = torch.mm(a, y)

        # concat feature vector and attention vector
        emb = torch.cat((out, att_vec), dim=1)

        # use label embedding to predict the labels
        emb_pre = self.mlp(emb)
        #print('prediction from embedding')
        #print(emb_pre[0])
        #print(emb_pre.shape)
        # try padding in label embedding and use the same mlp as emb
        pad = torch.zeros(y.shape[0], emb.shape[1] - y.shape[1])
        y = torch.hstack((pad, y))
        y_pre = self.mlp(y)
        #############
        # y_pre = self.mlp1(y)
        #y_pre = self.mlp1(y)
        return torch.sigmoid(emb_pre), y_pre


def model_train(train_loader):
    un_lbl = torch.arange(0, G.y.shape[1])

    outs1 = []
    total_loss = 0
    for idx in train_loader:
        x = G.x[idx]
        y = G.lbl_emb
        out1, out2 = model.forward(x, y)
        #print(out1.shape)
        outs1.append(out1)

        loss_train = BCE_loss(out1, G.y[idx]) + F.cross_entropy(out2, un_lbl)
        loss_train.backward()
        optimizer.step()

        total_loss += float(loss_train) * len(idx)

    output1 = torch.cat(outs1, dim=0)
    #print('output 0-3s')
    #print(output1[0:3])
    micro_train, macro_train = f1_loss(G.y[G.train_mask], output1)
    roc_auc_train_macro = _eval_rocauc(G.y[G.train_mask], output1)

    ap_train = ap_score(G.y[G.train_mask], output1)

    return total_loss/G.num_nodes, micro_train, macro_train, roc_auc_train_macro, ap_train

@torch.no_grad()
def model_test(data_loader):
    un_lbl = torch.arange(0, G.y.shape[1])
    # all embedding output
    outs1 = []
    for idx in data_loader:
        x = G.x[idx]
        y = G.lbl_emb
        out1, output2 = model.forward(x, y)
        outs1.append(out1)

    # embedding prediction
    output1 = torch.cat(outs1, dim=0)
    # calculate loss
    loss_val = BCE_loss(output1[G.val_mask], G.y[G.val_mask]) + F.cross_entropy(output2, un_lbl)

    micro_val, macro_val = f1_loss(G.y[G.val_mask], output1[G.val_mask])
    roc_auc_val_macro = _eval_rocauc(G.y[G.val_mask], output1[G.val_mask])
    ap_val = ap_score(G.y[G.val_mask], output1[G.val_mask])

    micro_test, macro_test = f1_loss(G.y[G.test_mask], output1[G.test_mask])
    roc_auc_test_macro = _eval_rocauc(G.y[G.test_mask], output1[G.test_mask])
    ap_test = ap_score(G.y[G.test_mask], output1[G.test_mask])

    return loss_val, micro_val, macro_val, roc_auc_val_macro, ap_val,\
           micro_test, macro_test, roc_auc_test_macro, ap_test


if __name__ == "__main__":
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        device = args.device_name

    if args.data_name in ["blogcatalog", "flickr", "youtube"]:
        G = load_mat_data(args.data_name, args.split_name, args.train_percent)
    elif args.data_name == "pcg_removed_isolated_nodes":
        G = load_pcg(args.data_name, args.split_name, args.train_percent)
    elif args.data_name == "Humloc":
        G = load_humloc()
    elif args.data_name == "Eukloc":
        G = load_eukloc()
    elif args.data_name == "dblp":
        G = load_dblp(args.data_name, args.split_name, args.train_percent)

    train_loader = DataLoader(G.n_id[G.train_mask],
                              shuffle=False,
                              batch_size=64,
                              num_workers=0)
    eva_loader = DataLoader(G.n_id,
                            shuffle=False,
                            batch_size=64,
                            num_workers=0)

    model = LANC(in_channels=G.x.shape[2],
                 class_channels=G.y.shape[1],
                 num_label=G.lbl_emb.shape[0])

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    early_stopping = EarlyStopping(data_name=args.data_name, split_name=args.split_name,
                                   patience=args.patience, verbose=True)

    for epoch in range(1, args.epochs):

        loss_train, micro_train, macro_train, roc_auc_train_macro, ap_train = model_train(train_loader)
        loss_val, micro_val, macro_val, roc_auc_val_macro, ap_val, \
        micro_test, macro_test, roc_auc_test_macro, ap_test = model_test(eva_loader)
        print(f'Epoch: {epoch:03d}, Loss: {loss_train:.10f}, '
              f'Train micro: {micro_train:.4f}, Train macro: {macro_train:.4f}, '
              f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f}, '
              f'train ROC-AUC macro: {roc_auc_train_macro:.4f} '
              f'val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
              f'test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
              f'Test Average Precision Score: {ap_test:.4f}, '
              )
        early_stopping(loss_val, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("Optimization Finished!")
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(args.data_name + args.split_name[:-3] + '_checkpoint.pt'))
    loss_val, micro_val, macro_val, roc_auc_val_macro, ap_val,\
    micro_test, macro_test, roc_auc_test_macro, ap_test = model_test(eva_loader)

    print(f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
          #f'train ROC-AUC macro: {roc_auc_train_macro:.4f} '
          #f'val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
          f'test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
          f'Test Average Precision Score: {ap_test:.4f}, '
          )





import argparse
import os.path as osp
import scipy
import torch
import torch.nn.functional as F
import scipy.io
import os
import torch_geometric.transforms as T
#from torch_geometric.logging import init_wandb, log
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import average_precision_score
from earlystopping import EarlyStopping
from torch_sparse import SparseTensor
from metric.metrics import f1_loss, BCE_loss, _eval_rocauc, ap_score


parser = argparse.ArgumentParser()
parser.add_argument("--data_name", default='blogcatalog',
                        help='Name of the dataset'
                        'Hyperspheres_64_64_0'
                        'pcg_removed_isolated_nodes'
                        'Humloc'
                        'yelp'
                        'ogbn-proteins'
                        'dblp')
parser.add_argument("--split_name", default='split_0.pt',
                        help='Name of the split')
parser.add_argument('--train_percent', type=float, default=0.6,
                        help='percentage of data used for training')
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
parser.add_argument('--wandb', action='store_true', help='Track experiment')
parser.add_argument('--patience', type=float, default=100,
                        help='patience for early stopping.')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 loss on parameters).')
args = parser.parse_args()


device = "cpu"


def normalize_tensor(a):
    """Row-normalize tensor that requires grad"""
    rowsum = a.sum(1).detach().cpu().numpy()
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(a.detach().cpu().numpy())
    mx = torch.tensor(mx)
    return mx


def row_normlize_sparsetensor(a):
    deg = a.to_dense().sum(dim=1).to(torch.float)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    a_n = deg_inv.view(-1, 1) * a
    return a_n


def glorot(shape):
    init_range = np.sqrt(6.0 / np.sum(shape))
    #(r1 - r2) * torch.rand(a, b) + r2
    initial = (init_range-init_range) * torch.rand(shape) + (-init_range)

    return initial


def normalize_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_hyper_data(data_name, split_name, train_percent, path):
    print('Loading dataset ' + data_name + '.csv...')

    labels = np.genfromtxt(osp.join(path, data_name, "labels.csv"),
                           skip_header=1, dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()

    features = np.genfromtxt(osp.join(path, data_name, "features.csv"),
                             skip_header=1, dtype=np.dtype(float), delimiter=',')

    features = torch.tensor(features).float()
    #features = features[:, :115]

    edge_index = torch.tensor(np.genfromtxt(osp.join(path, data_name, "edge_index.csv"),
                              dtype=np.dtype(float), delimiter=',')).long()
    # add self-loop
    edge_index = add_self_loops(edge_index)[0]
    #adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), size=(features.shape[0], features.shape[0]))
    #adj_t = normalize_tensor(adj.to_dense())

    folder_name = data_name + "_" + str(train_percent)
    file_path = osp.join(path, folder_name, split_name)
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

    y = labels.clone().detach().float()
    y[val_mask] = torch.zeros(labels.shape[1])
    y[test_mask] = torch.zeros(labels.shape[1])

    #num_class = labels.shape[1]
    num_nodes = labels.shape[0]

    # cnt_id = torch.unique(torch.flatten(edge_index)).tolist()
    # iso_mask = torch.ones(features.shape[0], dtype=torch.bool)
    # iso_mask[cnt_id] = False

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)
    #G.adj_t = adj_t
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    G.soft_labels = y
    G.n_id = torch.arange(num_nodes)

    edge_weights = glorot(shape=G.edge_index.shape[1])
    G.edge_weights = edge_weights
    return G


def load_mat_data(data_name, split_name, train_percent, path="../../data/"):
    print('Loading dataset ' + data_name + '.mat...')
    mat = scipy.io.loadmat(path + data_name)
    labels = mat['group']
    labels = sparse_mx_to_torch_sparse_tensor(labels).to_dense()

    adj = mat['network']
    adj = sparse_mx_to_torch_sparse_tensor(adj).long()
    edge_index = torch.transpose(torch.nonzero(adj.to_dense()), 0, 1).long()
    edge_index = add_self_loops(edge_index)[0]
    # prepare the feature matrix
    #features = torch.range(0, labels.shape[0] - 1).long()
    #features = torch.rand(labels.shape[0], 128)
    features = torch.eye(labels.shape[0])

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

    y = labels.clone().detach().float()
    y[val_mask] = torch.zeros(labels.shape[1])
    y[test_mask] = torch.zeros(labels.shape[1])

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    G.soft_labels = y
    G.n_id = torch.arange(num_nodes)
    edge_weights = glorot(shape=G.edge_index.shape[1])
    G.edge_weights = edge_weights

    return G


G = load_mat_data(data_name=args.data_name, split_name=args.split_name,
                  train_percent=args.train_percent, path="../../data/")


class GCN_LPA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_gcn, lpa_iter):
        super().__init__()

        self.num_gcn = num_gcn
        # default: add self loop, normalize
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_gcn):
            if i == 0:
                self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
            elif i == self.num_gcn-1:
                self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))
            else:
                self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=False))

        self.lpa_iter = lpa_iter
        self.edge_attr = torch.nn.Parameter(G.edge_weights.abs(), requires_grad=True)

    def forward(self, x, soft_labels, edge_index):
        weighted_adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                                    value=self.edge_attr, sparse_sizes=(x.shape[0], x.shape[0]))
        weighted_adj = row_normlize_sparsetensor(weighted_adj).to(device)

        #gcn

        x = F.dropout(x, p=0.5, training=self.training)
        #print("x", x)
        #print(self.edge_attr)

        x = F.relu(self.convs[0](x, weighted_adj))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, weighted_adj)
        m = torch.nn.Sigmoid()
        x = m(x)

        #lpa
        predicted_labels = soft_labels

        #print("edge_index", edge_index)
        #print("edge_attr", self.edge_attr)
        print("weighted adj", weighted_adj[:10])
        for i in range(self.lpa_iter):
            predicted_labels = torch.sparse.mm(weighted_adj.to_dense(), predicted_labels)
        #print(predicted_labels)
        predicted_labels = m(predicted_labels)
        #print(predicted_labels)
        return x, predicted_labels


model = GCN_LPA(in_channels=G.x.shape[1],
                hidden_channels=args.hidden_channels,
                out_channels=G.y.shape[1],
                num_gcn=2,
                lpa_iter=5,
                )
model.to(device)
G = G.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01,  weight_decay=args.weight_decay)


def model_train():
    model.train()
    optimizer.zero_grad()
    output, predicted_labels = model(G.x, G.soft_labels, G.edge_index)

    #gcn_loss
    Loss = torch.nn.BCELoss()

    gcn_loss = Loss(output[G.train_mask], G.y[G.train_mask])

    lpa_loss = Loss(predicted_labels[G.train_mask], G.y[G.train_mask])

    loss = gcn_loss + lpa_loss
    loss.backward()

    optimizer.step()
    return float(loss)


@torch.no_grad()
def model_test():
    model.eval()
    output, predicted_labels = model(G.x, G.soft_labels, G.edge_index)

    micro_train, macro_train = f1_loss(G.y[G.train_mask], output[G.train_mask])
    roc_auc_train_macro = _eval_rocauc(G.y[G.train_mask], output[G.train_mask])
    ap_train = ap_score(G.y[G.train_mask], output[G.train_mask])

    Loss = torch.nn.BCELoss()
    gcn_loss_val = Loss(output[G.val_mask], G.y[G.val_mask])
    lpa_loss_val = Loss(predicted_labels[G.val_mask], G.y[G.val_mask])
    loss_val = gcn_loss_val + lpa_loss_val

    micro_val, macro_val = f1_loss(G.y[G.val_mask], output[G.val_mask])
    roc_auc_val_macro = _eval_rocauc(G.y[G.val_mask], output[G.val_mask])
    ap_val = ap_score(G.y[G.val_mask], output[G.val_mask])

    micro_test, macro_test = f1_loss(G.y[G.test_mask], output[G.test_mask])
    roc_auc_test_macro = _eval_rocauc(G.y[G.test_mask], output[G.test_mask])
    ap_test = ap_score(G.y[G.test_mask], output[G.test_mask])

    return micro_train, macro_train, roc_auc_train_macro, ap_train, loss_val, micro_val, macro_val, roc_auc_val_macro, ap_val, micro_test, macro_test, roc_auc_test_macro, ap_test


early_stopping = EarlyStopping(patience=args.patience, verbose=True)
best_val_acc = final_test_acc = 0
for epoch in range(1, args.epochs + 1):
    loss_train = model_train()

    micro_train, macro_train, roc_auc_train_macro, ap_train, loss_val, micro_val, macro_val, roc_auc_val_macro, ap_val, micro_test, macro_test, roc_auc_test_macro, ap_test = model_test()
    print(f'Epoch: {epoch:03d}, Loss: {loss_train:.10f}, '
          f'Train micro: {micro_train:.4f}, Train macro: {macro_train:.4f},'
          f'Val micro: {micro_val:.4f}, Val macro: {macro_val:.4f}, '
          f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f}, '
          # f'Val ROC-AUC micro: {roc_auc_val_micro:.4f}, '
          f'val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
          # f'Test ROC-AUC micro: {roc_auc_test_micro:.4f}, '
          f'test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
          f'Test Average Precision Score: {ap_test:.4f}, '
          )
    early_stopping(loss_val, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

print("Optimization Finished!")
# load the last checkpoint with the best model
model.load_state_dict(torch.load('checkpoint.pt'))
micro_train, macro_train, roc_auc_train_macro, ap_train, loss_val, micro_val, macro_val, roc_auc_val_macro, ap_val, micro_test, macro_test, roc_auc_test_macro, ap_test = model_test()
print(f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
        #f'Val ROC-AUC micro: {roc_auc_val_micro:.4f}, '
         f'val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
        #f'Test ROC-AUC micro: {roc_auc_test_micro:.4f}, '
        f'test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
        f'Test Average Precision Score: {ap_test:.4f}, '
        )
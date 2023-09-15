import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch.optim as optim
import argparse
from torch_geometric.data import Data
from torch_geometric.datasets import Yelp
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv
import os
from earlystopping import EarlyStopping
from metric.metrics import f1_loss, BCE_loss, _eval_rocauc, ap_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default='yelp',
                        help='Name of the dataset:'
                             'Humloc')
    parser.add_argument('--train_percent', type=float, default=0.6,
                        help='percentage of data used for training')
    parser.add_argument("--split_name", default='split_2.pt',
                        help='Name of the split')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument("--device_name", default='cuda',
                        help='Name of the device')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--patience', type=float, default=100,
                        help='patience for early stopping.')
    return parser.parse_args()


args = parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = args.device_name


def import_yelp(data_name, split_name, train_percent, path="../../data/"):
    print('Loading dataset ' + data_name + '...')
    dataset = Yelp(root='../data/Yelp')
    data = dataset[0]
    labels = data.y
    features = data.x

    edge_index = data.edge_index
    adj = SparseTensor(row=edge_index[1], col=edge_index[0],
                       value=None,
                       sparse_sizes=(data.x.shape[0], data.x.shape[0]),
                       trust_data=True)

    adj_t = adj.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    data.adj_t = adj_t
    print(adj_t)
    print(type(adj_t))

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

    G = Data(x=features,
             y=labels)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    G.adj_t = adj_t
    G.n_id = torch.arange(G.x.shape[0])

    return G


G = import_yelp(args.data_name, args.split_name, args.train_percent)


class H2GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(H2GCN, self).__init__()
        # input
        self.dense1 = torch.nn.Linear(nfeat, nhid)
        # output
        self.dense2 = torch.nn.Linear(nhid*7, nclass)
        # drpout
        # self.dropout = SparseDropout(dropout)
        self.dropout = dropout
        # conv
        self.conv1 = GCNConv(nhid, nhid, normalize=False)
        self.conv2 = GCNConv(nhid*2, nhid*2, normalize=False)
        self.relu = torch.nn.ReLU()
        self.vec = torch.nn.Flatten()
        self.iden = torch.sparse.Tensor()

    def forward(self, features, edge_index):

        # feature space ----> hidden
        # adj2 = adj * adj
        # r1: compressed feature matrix
        x = self.relu(self.dense1(features))
        # # vectorize
        # x = self.vec(x)
        # aggregate info from 1 hop away neighbor
        # r2 torch.cat(x, self.conv(x, adj), self.conv(x, adj2))
        x11 = self.conv1(x, edge_index)
        x12 = self.conv1(x11, edge_index)
        x1 = torch.cat((x11, x12), -1)

        # vectorize
        # x = self.vec(x1)
        # aggregate info from 2 hp away neighbor
        x21 = self.conv2(x1, edge_index)
        x22 = self.conv2(x21, edge_index)
        x2 = torch.cat((x21, x22), -1)

        # concat
        x = torch.cat((x, x1, x2), dim=-1)
        # x = self.dropout(x)
        x = F.dropout(x, self.dropout)
        x = self.dense2(x)

        return F.sigmoid(x)


model = H2GCN(G.x.shape[1], args.hidden, G.y.shape[1])


optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay
                       )

model.to(device)
x = G.x.to(device)
labels = G.y.to(device)
edge_index = G.adj_t.to(device)
train_mask = G.train_mask.to(device)
val_mask = G.val_mask.to(device)
test_mask = G.test_mask.to(device)

early_stopping = EarlyStopping(patience=args.patience, verbose=True)


def model_train():
    model.train()
    optimizer.zero_grad()

    output = model(x, edge_index)
    loss_train = BCE_loss(output[train_mask], labels[train_mask])

    micro_train, macro_train = f1_loss(labels[train_mask], output[train_mask])
    roc_auc_train_macro = _eval_rocauc(labels[train_mask], output[train_mask])

    loss_train.backward()
    optimizer.step()

    return loss_train, micro_train, macro_train, roc_auc_train_macro


@torch.no_grad()
def model_test():
    model.eval()

    output = model(x, edge_index)

    loss_val = BCE_loss(output[val_mask], labels[val_mask])
    micro_val, macro_val = f1_loss(labels[val_mask], output[val_mask])
    roc_auc_val_macro = _eval_rocauc(labels[val_mask], output[val_mask])

    micro_test, macro_test = f1_loss(labels[test_mask], output[test_mask])
    roc_auc_test_macro = _eval_rocauc(labels[test_mask], output[test_mask])
    ap_test = ap_score(labels[test_mask], output[test_mask])

    return loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, ap_test


for epoch in range(1, args.epochs):
    loss_train, micro_train, macro_train, roc_auc_train_macro = model_train()
    loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, test_ap = model_test()
    print(f'Epoch: {epoch:03d}, Loss: {loss_train:.10f}, '
          f'Train micro: {micro_train:.4f}, Train macro: {macro_train:.4f} '
          f'Val micro: {micro_val:.4f}, Val macro: {macro_val:.4f} '
          f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
          f'train ROC-AUC macro: {roc_auc_train_macro:.4f} '
          f'Val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
          f'Test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
          f'Test Average Precision Score: {test_ap:.4f}, '
          )
    early_stopping(loss_val, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

print("Optimization Finished!")
# load the last checkpoint with the best model
model.load_state_dict(torch.load('checkpoint.pt'))
loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, test_ap = model_test()
print(f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
      # f'Val ROC-AUC micro: {roc_auc_val_micro:.4f}, '
      f'val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
      # f'Test ROC-AUC micro: {roc_auc_test_micro:.4f}, '
      f'test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
      f'Test Average Precision Score: {test_ap:.4f}, '
      )













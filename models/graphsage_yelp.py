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
from torch_geometric.nn import SAGEConv
import os
from earlystopping import EarlyStopping
from torch_geometric.loader import NeighborLoader
import copy
from metric.metrics import f1_loss, BCE_loss, _eval_rocauc, ap_score
from data.load_data import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default='blogcatalog',
                        help='Name of the dataset'
                             'Hyperspheres_10_10_0'
                             'pcg_removed_isolated_nodes'
                             'Humloc'
                             'yelp'
                             'ogbn-proteins'
                             "dblp")
    parser.add_argument('--train_percent', type=float, default=0.6,
                        help='percentage of data used for training')
    parser.add_argument("--split_name", default='split_2.pt',
                        help='Name of the split')
    parser.add_argument("--model_name", default='SAGE_sup',
                        help='GCN, GAT, SAGE_sup, MLP, H2GCN')
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
    parser.add_argument('--hidden', type=int, default=256,
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


G = import_yelp(args.data_name, args.split_name, args.train_percent)


class SAGE_sup(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, class_channels, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, class_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return F.sigmoid(x)

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
            x_all = torch.cat(xs, dim=0)
        return F.sigmoid(x_all)


model = SAGE_sup(in_channels=G.x.shape[1],
                 hidden_channels=args.hidden,
                 class_channels=G.y.shape[1],
                 )

kwargs = {'batch_size': 1024, 'num_workers': 6, 'persistent_workers': True}
train_loader = NeighborLoader(G, input_nodes=G.train_mask,
                              num_neighbors=[25, 10], shuffle=True, **kwargs)

subgraph_loader = NeighborLoader(copy.copy(G), input_nodes=None,
                                 num_neighbors=[-1], shuffle=False, **kwargs)


optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay
                       )
print(model)
model.to(device)
x = G.x.to(device)
labels = G.y.to(device)
edge_index = G.edge_index.to(device)
train_mask = G.train_mask.to(device)
val_mask = G.val_mask.to(device)
test_mask = G.test_mask.to(device)

early_stopping = EarlyStopping(patience=args.patience, verbose=True)


def model_train():
    model.train()
    optimizer.zero_grad()

    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x.to(device), batch.edge_index.to(device))[:batch.batch_size]
        labels_target = batch.y[:batch.batch_size].to(device)
        loss = BCE_loss(out, labels_target)

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.batch_size

    return total_loss/G.x.shape[0]


@torch.no_grad()
def model_test():
    model.eval()

    output = model.inference(x, subgraph_loader)
    labels = G.y.to(output.device)
    loss_train = BCE_loss(output[train_mask], labels[train_mask])
    micro_train, macro_train = f1_loss(labels[train_mask], output[train_mask])
    roc_auc_train_macro = _eval_rocauc(labels[train_mask], output[train_mask])

    loss_val = BCE_loss(output[val_mask], labels[val_mask])
    micro_val, macro_val = f1_loss(labels[val_mask], output[val_mask])
    roc_auc_val_macro = _eval_rocauc(labels[val_mask], output[val_mask])

    micro_test, macro_test = f1_loss(labels[test_mask], output[test_mask])
    roc_auc_test_macro = _eval_rocauc(labels[test_mask], output[test_mask])
    ap_test = ap_score(labels[test_mask], output[test_mask])

    return loss_train, micro_train, macro_train, roc_auc_train_macro, loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, ap_test


for epoch in range(1, args.epochs):
    loss_train = model_train()
    _, micro_train, macro_train, roc_auc_train_macro, loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, test_ap = model_test()
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
_, micro_train, macro_train, roc_auc_train_macro, loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, test_ap = model_test()
print(f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
      # f'Val ROC-AUC micro: {roc_auc_val_micro:.4f}, '
      f'val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
      # f'Test ROC-AUC micro: {roc_auc_test_micro:.4f}, '
      f'test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
      f'Test Average Precision Score: {test_ap:.4f}, '
      )













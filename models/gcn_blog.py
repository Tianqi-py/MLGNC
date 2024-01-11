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
import scipy.sparse as sp
import os
from earlystopping import EarlyStopping
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import scipy.io


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default='blogcatalog',
                        help='Name of the dataset:'
                             'Humloc')
    parser.add_argument('--train_percent', type=float, default=0.6,
                        help='percentage of data used for training')
    parser.add_argument("--split_name", default='split_2.pt',
                        help='Name of the split')
    parser.add_argument("--model_name", default='GCN',
                        help='GCN, GAT, SAGE_sup, MLP, H2GCN')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument("--device_name", default='cuda',
                        help='Name of the device')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=2000,
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


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def f1_loss(y, predictions):
    y = y.data.cpu().numpy()
    predictions = predictions.data.cpu().numpy()
    number_of_labels = y.shape[1]
    # find the indices (labels) with the highest probabilities (ascending order)
    pred_sorted = np.argsort(predictions, axis=1)

    # the true number of labels for each node
    num_labels = np.sum(y, axis=1)
    # we take the best k label predictions for all nodes, where k is the true number of labels
    pred_reshaped = []
    for pr, num in zip(pred_sorted, num_labels):
        pred_reshaped.append(pr[-int(num):].tolist())

    # convert back to binary vectors
    pred_transformed = MultiLabelBinarizer(classes=range(number_of_labels)).fit_transform(pred_reshaped)
    f1_micro = f1_score(y, pred_transformed, average='micro')
    f1_macro = f1_score(y, pred_transformed, average='macro')
    return f1_micro, f1_macro


def BCE_loss(outputs: torch.Tensor, labels: torch.Tensor):
    loss = torch.nn.BCELoss()
    bce = loss(outputs, labels)
    return bce


def _eval_rocauc(y_true, y_pred):
    '''
        compute ROC-AUC and AP score averaged across tasks
    '''

    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    rocauc_list = []

    for i in range(y_true.shape[1]):

        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            rocauc_list.append(roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    #return {'rocauc': sum(rocauc_list)/len(rocauc_list)}
    return sum(rocauc_list) / len(rocauc_list)


def ap_score(y_true, y_pred):

    ap_score = average_precision_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

    return ap_score


def model_train():
    model.train()
    optimizer.zero_grad()

    output = model(x, G.edge_index)
    loss_train = BCE_loss(output[train_mask], labels[train_mask])

    micro_train, macro_train = f1_loss(labels[train_mask], output[train_mask])
    roc_auc_train_macro = _eval_rocauc(labels[train_mask], output[train_mask])

    loss_train.backward()
    optimizer.step()

    return loss_train, micro_train, macro_train, roc_auc_train_macro


@torch.no_grad()
def model_test():
    model.eval()

    output = model(x, G.edge_index)

    loss_val = BCE_loss(output[val_mask], labels[val_mask])
    micro_val, macro_val = f1_loss(labels[val_mask], output[val_mask])
    roc_auc_val_macro = _eval_rocauc(labels[val_mask], output[val_mask])

    micro_test, macro_test = f1_loss(labels[test_mask], output[test_mask])
    roc_auc_test_macro = _eval_rocauc(labels[test_mask], output[test_mask])
    ap_test = ap_score(labels[test_mask], output[test_mask])

    return loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, ap_test


args = parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = args.device_name


def load_mat_data(data_name, split_name, train_percent, path="../../data/"):

    print('Loading dataset ' + data_name + '.mat...')
    mat = scipy.io.loadmat(path + data_name)
    labels = mat['group']
    labels = sparse_mx_to_torch_sparse_tensor(labels).to_dense()

    adj = mat['network']
    adj = sparse_mx_to_torch_sparse_tensor(adj).long()
    edge_index = torch.transpose(torch.nonzero(adj.to_dense()), 0, 1).long()
    # prepare the feature matrix
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

    return G


G = load_mat_data(data_name=args.data_name, split_name=args.split_name,
                  train_percent=args.train_percent, path="../data/")



class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, class_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, class_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.sigmoid(x)


model = GCN(in_channels=G.x.shape[1],
            hidden_channels=args.hidden,
            class_channels=G.y.shape[1],
            )


optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay
                       )

model.to(device)
x = G.x.to(device)
labels = G.y.to(device)
train_mask = G.train_mask.to(device)
val_mask = G.val_mask.to(device)
test_mask = G.test_mask.to(device)

early_stopping = EarlyStopping(patience=args.patience, verbose=True)


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
# loss_val, micro_val, macro_val, roc_auc_val_micro, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_micro, roc_auc_test_macro, test_ap = model_test()
loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, test_ap = model_test()
print(f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
        # f'Val ROC-AUC micro: {roc_auc_val_micro:.4f}, '
        f'val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
        # f'Test ROC-AUC micro: {roc_auc_test_micro:.4f}, '
        f'test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
        f'Test Average Precision Score: {test_ap:.4f}, '
        )













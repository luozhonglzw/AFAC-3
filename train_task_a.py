"""
Task A: Graph Node Classification with GCN
Handles: directed graph, isolated nodes, class imbalance
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix, diags
from sklearn.model_selection import train_test_split

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_data():
    npz_path = os.path.join(DATA_ROOT, "A分类", "A分类", "A1.npz")
    data = np.load(npz_path, allow_pickle=True)

    adj = csr_matrix(
        (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
        shape=tuple(data["adj_shape"]),
    )
    features = csr_matrix(
        (data["attr_data"], data["attr_indices"], data["attr_indptr"]),
        shape=tuple(data["attr_shape"]),
    )
    labels = data["labels"]
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]

    return adj, features, labels, train_idx, test_idx


# ─────────────────────────────────────────────
# Graph Preprocessing
# ─────────────────────────────────────────────

def normalize_adj_symmetric(adj):
    """D^{-1/2} A D^{-1/2} with self-loops (GCN-style)"""
    adj = adj + csr_matrix(np.eye(adj.shape[0]), dtype=np.float32)  # add self-loops
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = diags(deg_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
    return adj_norm


def adj_to_sparse_tensor(adj):
    """Convert scipy sparse matrix to PyTorch sparse tensor"""
    adj = adj.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64))
    values = torch.from_numpy(adj.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(adj.shape))


def prepare_data():
    adj, features, labels, train_idx, test_idx = load_data()

    # Symmetrize the directed graph for better message passing
    adj_sym = adj + adj.T
    adj_sym.data = np.ones_like(adj_sym.data)  # binarize

    # Normalize
    adj_norm = normalize_adj_symmetric(adj_sym)

    # Convert to tensors
    adj_tensor = adj_to_sparse_tensor(adj_norm).to(DEVICE)
    feat_tensor = torch.from_numpy(features.toarray().astype(np.float32)).to(DEVICE)
    labels_tensor = torch.from_numpy(labels.astype(np.int64)).to(DEVICE)

    # Split train into train/val (80/20)
    train_idx_arr = np.array(train_idx)
    train_sub, val_sub = train_test_split(
        train_idx_arr, test_size=0.2, random_state=42,
        stratify=labels[train_idx_arr]
    )
    train_mask = torch.zeros(len(labels), dtype=torch.bool, device=DEVICE)
    val_mask = torch.zeros(len(labels), dtype=torch.bool, device=DEVICE)
    test_mask = torch.zeros(len(labels), dtype=torch.bool, device=DEVICE)
    train_mask[train_sub] = True
    val_mask[val_sub] = True
    test_mask[test_idx] = True

    # Class weights for imbalanced data
    train_labels = labels[train_sub]
    class_counts = np.bincount(train_labels, minlength=10)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * 10
    class_weights_tensor = torch.from_numpy(class_weights.astype(np.float32)).to(DEVICE)

    print(f"Data loaded: {feat_tensor.shape[0]} nodes, {feat_tensor.shape[1]} features")
    print(f"Train: {train_sub.__len__()}, Val: {val_sub.__len__()}, Test: {len(test_idx)}")
    print(f"Class weights: {class_weights.round(3)}")

    return adj_tensor, feat_tensor, labels_tensor, train_mask, val_mask, test_mask, class_weights_tensor, test_idx


# ─────────────────────────────────────────────
# GCN Model
# ─────────────────────────────────────────────

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None

    def forward(self, x, adj_norm):
        # adj_norm @ x @ weight
        support = x @ self.weight
        out = torch.sparse.mm(adj_norm, support)
        if self.bias is not None:
            out = out + self.bias
        return out


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout

        if num_layers == 1:
            self.layers.append(GCNLayer(in_dim, out_dim))
        else:
            self.layers.append(GCNLayer(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(GCNLayer(hidden_dim, hidden_dim))
            self.layers.append(GCNLayer(hidden_dim, out_dim))

    def forward(self, x, adj_norm):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adj_norm)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, adj_norm)
        return x


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_epoch(model, adj, feats, labels, mask, optimizer, class_weights):
    model.train()
    optimizer.zero_grad()
    out = model(feats, adj)
    loss = F.cross_entropy(out[mask], labels[mask], weight=class_weights)
    loss.backward()
    optimizer.step()

    pred = out[mask].argmax(dim=1)
    acc = (pred == labels[mask]).float().mean().item()
    return loss.item(), acc


@torch.no_grad()
def evaluate(model, adj, feats, labels, mask):
    model.eval()
    out = model(feats, adj)
    pred = out[mask].argmax(dim=1)
    acc = (pred == labels[mask]).float().mean().item()
    return acc


def run_training():
    adj, feats, labels, train_mask, val_mask, test_mask, class_weights, test_idx = prepare_data()

    # Hyperparameters
    HIDDEN_DIM = 256
    NUM_LAYERS = 3
    DROPOUT = 0.5
    LR = 0.01
    WEIGHT_DECAY = 5e-4
    EPOCHS = 300
    PATIENCE = 50

    model = GCN(
        in_dim=feats.shape[1],
        hidden_dim=HIDDEN_DIM,
        out_dim=10,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: GCN-{NUM_LAYERS}, hidden={HIDDEN_DIM}, params={num_params}")
    print(f"Training on {DEVICE}...")

    best_val_acc = 0
    patience_counter = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, adj, feats, labels, train_mask, optimizer, class_weights)
        val_acc = evaluate(model, adj, feats, labels, val_mask)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={train_loss:.4f} | "
                  f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.5f} | {time.time()-t0:.1f}s")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Load best model and predict on test
    model.load_state_dict(best_state)
    model.to(DEVICE)

    test_acc_on_val = evaluate(model, adj, feats, labels, val_mask)
    print(f"\nBest val acc: {best_val_acc:.4f}")

    # Generate test predictions
    model.eval()
    with torch.no_grad():
        out = model(feats, adj)
        test_pred = out[test_idx].argmax(dim=1).cpu().numpy()

    # Save submission
    sub_path = os.path.join(DATA_ROOT, "A1.csv")
    sample = pd.read_csv(os.path.join(DATA_ROOT, "A分类", "A分类", "sample_submission.csv"))
    sample["label"] = test_pred
    sample.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
    print(f"Prediction distribution: {np.bincount(test_pred, minlength=10)}")


if __name__ == "__main__":
    run_training()

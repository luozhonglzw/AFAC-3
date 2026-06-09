"""APPNP: Predict then Propagate - grid search K and alpha"""

import os
import time
import json
import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F


class APPNP(nn.Module):
    """APPNP: feature transform first, then propagate with teleport"""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, K=10, alpha=0.1, dropout=0.5):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x, adj):
        # Feature transformation (no propagation in hidden layers)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        H0 = self.layers[-1](x)  # (N, out_dim)

        # APPNP propagation: H^(k) = (1-alpha) * A @ H^(k-1) + alpha * H0
        H = H0
        for _ in range(self.K):
            H = (1 - self.alpha) * torch.sparse.mm(adj, H) + self.alpha * H0

        return H


def sparse_symmetric_norm(adj):
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)
    adj = adj + csr_matrix(np.eye(adj.shape[0]), dtype=np.float32)
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D = diags(deg_inv_sqrt)
    adj_norm = D @ adj @ D
    adj_norm = adj_norm.tocoo().astype(np.float32)
    idx = torch.from_numpy(np.vstack((adj_norm.row, adj_norm.col)).astype(np.int64))
    val = torch.from_numpy(adj_norm.data)
    return torch.sparse_coo_tensor(idx, val, torch.Size(adj_norm.shape))


def train_appnp(K, alpha):
    from train_cls import load_data

    adj, features, labels, train_idx, test_idx = load_data()

    config = {
        "hidden_dim": 256,
        "num_layers": 2,
        "K": K,
        "alpha": alpha,
        "dropout": 0.5,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "epochs": 300,
        "patience": 50,
    }

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feat = StandardScaler().fit_transform(features.toarray())
    feat_t = torch.from_numpy(feat.astype(np.float32)).to(DEVICE)
    adj_t = sparse_symmetric_norm(adj).to(DEVICE)
    labels_t = torch.from_numpy(labels.astype(np.int64)).to(DEVICE)

    idx = np.array(train_idx)
    trn, val = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels[idx])
    trn_mask = torch.zeros(len(labels), dtype=torch.bool, device=DEVICE)
    val_mask = torch.zeros(len(labels), dtype=torch.bool, device=DEVICE)
    trn_mask[trn] = True
    val_mask[val] = True

    cnt = np.bincount(labels[trn], minlength=10)
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, cnt)
    w = (1.0 - beta) / (effective_num + 1e-6)
    w = w / w.sum() * 10
    w_t = torch.from_numpy(w.astype(np.float32)).to(DEVICE)

    model = APPNP(767, config["hidden_dim"], 10,
                  config["num_layers"], config["K"], config["alpha"], config["dropout"]).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config["epochs"])

    best_acc = 0
    patience_cnt = 0
    best_state = None

    for ep in range(1, config["epochs"] + 1):
        model.train()
        opt.zero_grad()
        out = model(feat_t, adj_t)
        loss = F.cross_entropy(out[trn_mask], labels_t[trn_mask], weight=w_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            out = model(feat_t, adj_t)
            pred = out[val_mask].argmax(1)
            acc = (pred == labels_t[val_mask]).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1

        if ep % 50 == 0 or ep == 1:
            print(f"  Ep {ep:3d} | loss={loss.item():.4f} | val_acc={acc:.4f}")

        if patience_cnt >= config["patience"]:
            print(f"  Early stop at epoch {ep}")
            break

    return best_acc, best_state, config


def main():
    results = []
    for K in [5, 10]:
        for alpha in [0.1, 0.2]:
            print(f"\nAPPNP K={K}, alpha={alpha}")
            acc, state, config = train_appnp(K, alpha)
            results.append((K, alpha, acc, state, config))
            print(f"  val_acc={acc:.4f}")

    best = max(results, key=lambda x: x[2])
    K, alpha, acc, state, config = best
    print(f"\nBest: K={K}, alpha={alpha}, val_acc={acc:.4f}")

    if acc > 0.7315:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            "model_state": state,
            "config": config,
            "best_val_acc": acc,
        }, os.path.join(save_dir, "appnp_best.pt"))
        print(f"APPNP saved (beats GCN 73.15%)")
    else:
        print(f"APPNP {acc:.4f} < GCN 73.15%, not saved")


if __name__ == "__main__":
    main()

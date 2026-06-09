"""Sparse GAT (Graph Attention Network) - computes attention on edges only"""

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


class SparseGATLayer(nn.Module):
    """Sparse GAT layer with edge-wise attention coefficients"""
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.6, concat=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.concat = concat
        self.dropout = dropout

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_l = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.a_r = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_l)
        nn.init.xavier_uniform_(self.a_r)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, edge_norm=None):
        """
        x: (N, in_dim) node features
        edge_index: (2, E) COO edge indices [src, dst]
        edge_norm: (E,) optional edge weights
        """
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        h = self.W(x).view(N, self.num_heads, self.head_dim)

        h_src = h[src]
        h_dst = h[dst]

        e_src = (h_src * self.a_l.unsqueeze(0)).sum(dim=-1)
        e_dst = (h_dst * self.a_r.unsqueeze(0)).sum(dim=-1)

        e = self.leaky_relu(e_src + e_dst)

        # sparse softmax over incoming edges per destination node
        e = torch.exp(e - e.max(dim=0, keepdim=True).values)
        if edge_norm is not None:
            e = e * edge_norm.unsqueeze(-1)

        denom = torch.zeros(N, self.num_heads, device=x.device, dtype=x.dtype)
        denom.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.num_heads), e)
        denom = denom.clamp(min=1e-10)
        alpha = e / denom[dst]

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = alpha.unsqueeze(-1) * h_src
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        dst_expanded = dst.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, self.head_dim)
        out.scatter_add_(0, dst_expanded, msg)

        if self.concat:
            return out.view(N, -1)
        else:
            return out.mean(dim=1)


class SparseGAT(nn.Module):
    """Sparse GAT network, same interface as GCN"""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, num_heads=4, dropout=0.6):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout

        self.layers.append(SparseGATLayer(in_dim, hidden_dim, num_heads, dropout, concat=True))
        for _ in range(num_layers - 2):
            self.layers.append(SparseGATLayer(hidden_dim, hidden_dim, num_heads, dropout, concat=True))
        self.layers.append(SparseGATLayer(hidden_dim, out_dim, 1, dropout, concat=False))

    def forward(self, x, adj_sparse):
        edge_index = adj_sparse.coalesce().indices()
        edge_norm = adj_sparse.coalesce().values()

        for layer in self.layers[:-1]:
            x = F.relu(layer(x, edge_index, edge_norm))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x, edge_index, edge_norm)


def sparse_symmetric_norm(adj):
    """Symmetric normalization: D^{-1/2} A D^{-1/2}"""
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


def train_gat():
    """Train GAT model"""
    from train_cls import load_data

    print("Loading data...")
    adj, features, labels, train_idx, test_idx = load_data()

    config = {
        "model_type": "GAT",
        "hidden_dim": 256,
        "num_layers": 3,
        "num_heads": 4,
        "dropout": 0.5,
        "lr": 0.005,
        "weight_decay": 5e-4,
        "epochs": 300,
        "patience": 50,
    }
    print(f"Config: {json.dumps(config, indent=2)}")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

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
    print(f"Class distribution: {cnt.tolist()}")
    print(f"Weights: {w.round(3).tolist()}")

    model = SparseGAT(767, config["hidden_dim"], 10,
                      config["num_layers"], config["num_heads"], config["dropout"]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GAT params: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config["epochs"])

    best_acc = 0
    patience_cnt = 0
    best_state = None

    print("\nTraining...")
    for ep in range(1, config["epochs"] + 1):
        t0 = time.time()
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

        if ep % 20 == 0 or ep == 1:
            lr = sched.get_last_lr()[0]
            print(f"  Ep {ep:3d} | loss={loss.item():.4f} | val_acc={acc:.4f} | lr={lr:.5f} | {time.time()-t0:.1f}s")

        if patience_cnt >= config["patience"]:
            print(f"  Early stop at epoch {ep}")
            break

    print(f"\nGAT best val accuracy: {best_acc:.4f}")

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save({
        "model_state": best_state,
        "config": config,
        "best_val_acc": best_acc,
    }, os.path.join(save_dir, "gat_best.pt"))
    print(f"Model saved to checkpoints/gat_best.pt")

    return best_acc


if __name__ == "__main__":
    train_gat()

"""GCN with graph structure features (degree, neighbor avg degree) appended"""

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

from train_cls import GCN, load_data, preprocess_graph, to_sparse_tensor


def compute_graph_features(adj):
    """Compute degree and neighbor average degree for each node"""
    N = adj.shape[0]

    # Degree
    deg = np.array(adj.sum(axis=1)).flatten().astype(np.float32)

    # Symmetrize for neighbor computation
    adj_sym = adj + adj.T
    adj_sym.data = np.ones_like(adj_sym.data, dtype=np.float32)

    # Neighbor average degree
    deg_sym = np.array(adj_sym.sum(axis=1)).flatten()
    neighbor_deg_sum = adj_sym @ deg_sym
    neighbor_avg_deg = np.zeros(N, dtype=np.float32)
    mask = deg_sym > 0
    neighbor_avg_deg[mask] = neighbor_deg_sum[mask] / deg_sym[mask]

    # Stack: (N, 2)
    graph_feats = np.column_stack([deg, neighbor_avg_deg])
    return graph_feats


def main():
    print("Loading data...")
    adj, features, labels, train_idx, test_idx = load_data()

    # Compute graph features
    print("Computing graph features...")
    graph_feats = compute_graph_features(adj)
    print(f"  Degree: mean={graph_feats[:,0].mean():.2f}, std={graph_feats[:,0].std():.2f}")
    print(f"  Neighbor avg deg: mean={graph_feats[:,1].mean():.2f}, std={graph_feats[:,1].std():.2f}")

    # Concatenate with original features
    orig_feat = features.toarray()
    feat = np.hstack([orig_feat, graph_feats])
    print(f"  Feature dim: {orig_feat.shape[1]} -> {feat.shape[1]}")

    config = {
        "model_type": "GCN",
        "hidden_dim": 256,
        "num_layers": 3,
        "dropout": 0.5,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "epochs": 300,
        "patience": 50,
        "feature_norm": True,
        "symmetrize": True,
        "norm_mode": "symmetric",
    }

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    if config["feature_norm"]:
        feat = StandardScaler().fit_transform(feat)
    feat_t = torch.from_numpy(feat.astype(np.float32)).to(DEVICE)

    adj_norm = preprocess_graph(adj, config["symmetrize"], config["norm_mode"])
    adj_t = to_sparse_tensor(adj_norm).to(DEVICE)
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

    model = GCN(769, config["hidden_dim"], 10, config["num_layers"], config["dropout"]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GCN params: {n_params:,}")

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

    print(f"\nGCN+feat best val accuracy: {best_acc:.4f}")

    if best_acc > 0.7315:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            "model_state": best_state,
            "config": config,
            "best_val_acc": best_acc,
            "feat_dim": 769,
        }, os.path.join(save_dir, "cls_feat_best.pt"))
        print(f"Model saved (beats GCN 73.15%)")
    else:
        print(f"GCN+feat {best_acc:.4f} < GCN 73.15%, not saved")

    return best_acc


if __name__ == "__main__":
    main()

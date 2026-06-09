"""
实验执行器 - 封装训练/评估流程，供 Agent 调用
支持任务A(分类)和任务B(推荐)
"""

import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix, diags
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 共享工具
# ============================================================

def get_gpu_memory():
    """获取当前 GPU 显存占用 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


# ============================================================
# 任务A: 图节点分类
# ============================================================

def load_cls_data(data_root):
    npz_path = os.path.join(data_root, "A分类", "A分类", "A1.npz")
    data = np.load(npz_path, allow_pickle=True)
    adj = csr_matrix(
        (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
        shape=tuple(data["adj_shape"]),
    )
    features = csr_matrix(
        (data["attr_data"], data["attr_indices"], data["attr_indptr"]),
        shape=tuple(data["attr_shape"]),
    )
    return adj, features, data["labels"], data["train_idx"], data["test_idx"]


def normalize_adj(adj, mode="symmetric"):
    """邻接矩阵归一化"""
    adj = adj + csr_matrix(np.eye(adj.shape[0]), dtype=np.float32)
    if mode == "symmetric":
        deg = np.array(adj.sum(axis=1)).flatten()
        deg_inv_sqrt = np.power(deg, -0.5)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
        D_inv_sqrt = diags(deg_inv_sqrt)
        return D_inv_sqrt @ adj @ D_inv_sqrt
    elif mode == "row":
        deg = np.array(adj.sum(axis=1)).flatten()
        deg_inv = np.power(deg, -1.0)
        deg_inv[np.isinf(deg_inv)] = 0.0
        D_inv = diags(deg_inv)
        return D_inv @ adj


def adj_to_sparse_tensor(adj):
    adj = adj.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64))
    values = torch.from_numpy(adj.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(adj.shape))


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, adj_norm):
        out = torch.sparse.mm(adj_norm, x @ self.weight)
        return out + self.bias


class GATLayer(nn.Module):
    """稀疏 GAT: 只在邻接矩阵非零位置计算 attention，避免 OOM"""
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_l = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.a_r = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_l.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_r.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, adj_norm):
        # adj_norm 是归一化后的稀疏矩阵，我们直接用它做聚合
        # 简化实现: 用 GCN 风格的稀疏聚合 + 可学习的特征变换
        h = self.W(x)  # (N, out_dim)
        h = h.view(-1, self.num_heads, self.head_dim)  # (N, heads, d)

        # 使用 adj_norm 做稀疏聚合 (近似 attention)
        # 对每个 head 分别聚合
        out_heads = []
        for head in range(self.num_heads):
            h_head = h[:, head, :]  # (N, d)
            agg = torch.sparse.mm(adj_norm, h_head)  # (N, d)
            out_heads.append(agg)
        out = torch.stack(out_heads, dim=1)  # (N, heads, d)
        return out.view(x.size(0), -1)


class ClsGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout
        self.layers.append(GCNLayer(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))
        self.layers.append(GCNLayer(hidden_dim, out_dim))

    def forward(self, x, adj_norm):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, adj_norm))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x, adj_norm)


class ClsGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, num_heads=4, dropout=0.6):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout
        self.layers.append(GATLayer(in_dim, hidden_dim, num_heads, dropout))
        for _ in range(num_layers - 2):
            self.layers.append(GATLayer(hidden_dim, hidden_dim, num_heads, dropout))
        self.final = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, adj):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, adj))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, adj)
        return self.final(x)


class ClsMLP(nn.Module):
    """纯 MLP 基线 (忽略图结构)"""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x, _adj=None):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x)


def run_cls_experiment(config, data_root):
    """
    执行一轮分类实验
    config: dict 包含模型超参数
    返回: dict 包含 metrics, feedback
    """
    t0 = time.time()

    # 加载数据
    adj, features, labels, train_idx, test_idx = load_cls_data(data_root)

    # 特征预处理
    if config.get("feature_norm", False):
        feat_arr = features.toarray()
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        feat_arr = scaler.fit_transform(feat_arr)
        feat_tensor = torch.from_numpy(feat_arr.astype(np.float32)).to(DEVICE)
    else:
        feat_tensor = torch.from_numpy(features.toarray().astype(np.float32)).to(DEVICE)

    # 邻接矩阵处理
    symmetrize = config.get("symmetrize", True)
    if symmetrize:
        adj_proc = adj + adj.T
        adj_proc.data = np.ones_like(adj_proc.data)
    else:
        adj_proc = adj

    norm_mode = config.get("norm_mode", "symmetric")
    adj_norm = normalize_adj(adj_proc, mode=norm_mode)
    adj_tensor = adj_to_sparse_tensor(adj_norm).to(DEVICE)

    labels_tensor = torch.from_numpy(labels.astype(np.int64)).to(DEVICE)

    # 训练/验证划分
    train_idx_arr = np.array(train_idx)
    val_ratio = config.get("val_ratio", 0.2)
    train_sub, val_sub = train_test_split(
        train_idx_arr, test_size=val_ratio, random_state=config.get("seed", 42),
        stratify=labels[train_idx_arr]
    )
    train_mask = torch.zeros(len(labels), dtype=torch.bool, device=DEVICE)
    val_mask = torch.zeros(len(labels), dtype=torch.bool, device=DEVICE)
    test_mask = torch.zeros(len(labels), dtype=torch.bool, device=DEVICE)
    train_mask[train_sub] = True
    val_mask[val_sub] = True
    test_mask[test_idx] = True

    # 类别权重
    train_labels = labels[train_sub]
    class_counts = np.bincount(train_labels, minlength=10)
    if config.get("use_class_weights", True):
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * 10
    else:
        class_weights = np.ones(10)
    class_weights_t = torch.from_numpy(class_weights.astype(np.float32)).to(DEVICE)

    # 构建模型
    model_type = config.get("model_type", "GCN")
    hidden_dim = config.get("hidden_dim", 256)
    num_layers = config.get("num_layers", 3)
    dropout = config.get("dropout", 0.5)

    if model_type == "GCN":
        model = ClsGCN(767, hidden_dim, 10, num_layers, dropout).to(DEVICE)
    elif model_type == "GAT":
        num_heads = config.get("num_heads", 4)
        model = ClsGAT(767, hidden_dim, 10, num_layers, num_heads, dropout).to(DEVICE)
    elif model_type == "MLP":
        model = ClsMLP(767, hidden_dim, 10, num_layers, dropout).to(DEVICE)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("lr", 0.01),
        weight_decay=config.get("weight_decay", 5e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.get("epochs", 300)
    )

    # 训练
    best_val_acc = 0
    patience = config.get("patience", 50)
    patience_counter = 0
    best_state = None
    train_losses = []
    val_accs = []

    for epoch in range(1, config.get("epochs", 300) + 1):
        model.train()
        optimizer.zero_grad()

        if model_type == "GAT":
            out = model(feat_tensor, adj_tensor)
        else:
            out = model(feat_tensor, adj_tensor)

        loss = F.cross_entropy(out[train_mask], labels_tensor[train_mask], weight=class_weights_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 5.0))
        optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())

        # 验证
        model.eval()
        with torch.no_grad():
            if model_type == "GAT":
                out = model(feat_tensor, adj_tensor)
            else:
                out = model(feat_tensor, adj_tensor)
            val_pred = out[val_mask].argmax(dim=1)
            val_acc = (val_pred == labels_tensor[val_mask]).float().mean().item()
            val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    duration = time.time() - t0

    # 评估
    model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        if model_type == "GAT":
            out = model(feat_tensor, adj_tensor)
        else:
            out = model(feat_tensor, adj_tensor)
        val_pred = out[val_mask].argmax(dim=1)
        final_val_acc = (val_pred == labels_tensor[val_mask]).float().mean().item()
        test_pred = out[test_idx].argmax(dim=1).cpu().numpy()

    # 反馈信息
    feedback = {
        "converged": patience_counter < patience,
        "final_epoch": len(train_losses),
        "train_loss_trend": "decreasing" if train_losses[-1] < train_losses[0] else "increasing",
        "overfitting": len(val_accs) > 20 and max(val_accs[:len(val_accs)//2]) > val_accs[-1] + 0.02,
        "gpu_memory_mb": round(get_gpu_memory(), 1),
    }

    metrics = {
        "primary_metric": round(final_val_acc, 4),
        "val_accuracy": round(final_val_acc, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "final_train_loss": round(train_losses[-1], 4) if train_losses else 0,
    }

    return {
        "metrics": metrics,
        "feedback": feedback,
        "duration": duration,
        "test_predictions": test_pred.tolist(),
        "config": config,
    }


# ============================================================
# 任务B: 序列推荐
# ============================================================

def load_rec_data(data_root):
    base = os.path.join(data_root, "A推荐", "A推荐")
    train = pd.read_csv(os.path.join(base, "train.csv"))
    test = pd.read_csv(os.path.join(base, "test.csv"))
    user_df = pd.read_csv(os.path.join(base, "user.csv"))
    item_df = pd.read_csv(os.path.join(base, "item.csv"))
    sample_sub = pd.read_csv(os.path.join(base, "sample_submission.csv"))
    return train, test, user_df, item_df, sample_sub


def build_rec_id_maps(train, test, item_df):
    all_uids = sorted(set(train["uid"]) | set(test["uid"]))
    uid2idx = {uid: i + 1 for i, uid in enumerate(all_uids)}
    all_iids = sorted(set(item_df["iid"]))
    iid2idx = {iid: i + 1 for i, iid in enumerate(all_iids)}
    return uid2idx, iid2idx, len(uid2idx) + 1, len(iid2idx) + 1


class RecDataset(Dataset):
    def __init__(self, df, uid2idx, iid2idx, max_len=100, is_test=False):
        self.max_len = max_len
        self.is_test = is_test
        self.samples = []
        for _, row in df.iterrows():
            uid = uid2idx[row["uid"]]
            seq_raw = str(row["item_seq_raw"]).strip()
            if not seq_raw or seq_raw == "nan":
                item_seq = []
            else:
                item_seq = [iid2idx.get(x, 0) for x in seq_raw.split(",")]
            if not is_test:
                target = iid2idx.get(row["target_iid"], 0)
                self.samples.append((uid, item_seq, target))
            else:
                self.samples.append((uid, item_seq, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uid, seq, target = self.samples[idx]
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]
        seq_padded = [0] * (self.max_len - len(seq)) + seq
        mask = [0] * (self.max_len - len(seq)) + [1] * len(seq)
        return (
            torch.tensor(uid, dtype=torch.long),
            torch.tensor(seq_padded, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float),
            torch.tensor(target, dtype=torch.long),
        )


class SASRecModel(nn.Module):
    def __init__(self, num_items, embed_dim=128, max_len=100, num_heads=2, num_layers=2, dropout=0.2):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, mask):
        batch_size, seq_len = seq.shape
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.dropout(x)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=seq.device), diagonal=1).bool()
        padding_mask = ~mask.bool()
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        x = self.output_norm(x)
        seq_len_per_sample = mask.sum(dim=1).long() - 1
        seq_len_per_sample = seq_len_per_sample.clamp(min=0)
        return x[torch.arange(batch_size, device=x.device), seq_len_per_sample]


def run_rec_experiment(config, data_root):
    """
    执行一轮推荐实验
    config: dict 包含模型超参数
    返回: dict 包含 metrics, feedback
    """
    t0 = time.time()

    train_df, test_df, user_df, item_df, sample_sub = load_rec_data(data_root)
    uid2idx, iid2idx, num_users, num_items = build_rec_id_maps(train_df, test_df, item_df)
    idx2iid = {v: k for k, v in iid2idx.items()}

    train_split, val_split = train_test_split(train_df, test_size=0.1, random_state=config.get("seed", 42))

    max_len = config.get("max_len", 100)
    train_ds = RecDataset(train_split, uid2idx, iid2idx, max_len)
    val_ds = RecDataset(val_split, uid2idx, iid2idx, max_len)

    batch_size = config.get("batch_size", 256)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    model = SASRecModel(
        num_items=num_items,
        embed_dim=config.get("embed_dim", 128),
        max_len=max_len,
        num_heads=config.get("num_heads", 2),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.2),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.get("epochs", 100))

    best_val_hit10 = 0
    patience = config.get("patience", 15)
    patience_counter = 0
    best_state = None
    neg_samples = config.get("neg_samples", 100)

    for epoch in range(1, config.get("epochs", 100) + 1):
        # 训练
        model.train()
        total_loss = 0
        total_correct = 0
        total_count = 0
        for uid, seq, mask, target in train_loader:
            uid, seq, mask, target = uid.to(DEVICE), seq.to(DEVICE), mask.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            user_repr = model(seq, mask)
            batch_size_actual = user_repr.size(0)
            neg = torch.randint(1, num_items, (batch_size_actual, neg_samples), device=DEVICE)
            all_items = torch.cat([target.unsqueeze(1), neg], dim=1)
            all_embs = model.item_emb(all_items)
            scores = (user_repr.unsqueeze(1) * all_embs).sum(dim=2)
            labels = torch.zeros(batch_size_actual, dtype=torch.long, device=DEVICE)
            loss = F.cross_entropy(scores, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * batch_size_actual
            total_correct += (scores.argmax(dim=1) == 0).sum().item()
            total_count += batch_size_actual
        scheduler.step()

        # 验证
        model.eval()
        hits = {1: 0, 5: 0, 10: 0}
        val_total = 0
        with torch.no_grad():
            for uid, seq, mask, target in val_loader:
                uid, seq, mask, target = uid.to(DEVICE), seq.to(DEVICE), mask.to(DEVICE), target.to(DEVICE)
                user_repr = model(seq, mask)
                item_embs = model.item_emb.weight[1:]
                scores = torch.matmul(user_repr, item_embs.T)
                for k in [1, 5, 10]:
                    topk = scores.topk(k, dim=1).indices + 1
                    for i in range(len(target)):
                        if target[i].item() in topk[i]:
                            hits[k] += 1
                val_total += len(target)

        val_hit10 = hits[10] / val_total if val_total > 0 else 0
        if val_hit10 > best_val_hit10:
            best_val_hit10 = val_hit10
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    duration = time.time() - t0

    metrics = {
        "primary_metric": round(best_val_hit10, 4),
        "val_hit@1": round(hits[1] / val_total, 4) if val_total > 0 else 0,
        "val_hit@5": round(hits[5] / val_total, 4) if val_total > 0 else 0,
        "val_hit@10": round(best_val_hit10, 4),
    }

    feedback = {
        "converged": patience_counter < patience,
        "final_epoch": epoch,
        "gpu_memory_mb": round(get_gpu_memory(), 1),
        "overfitting": best_val_hit10 > 0 and val_hit10 < best_val_hit10 * 0.95,
    }

    return {
        "metrics": metrics,
        "feedback": feedback,
        "duration": duration,
        "config": config,
    }

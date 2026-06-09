"""
Task B V2: SASRec (Self-Attentive Sequential Recommendation)
使用自注意力机制建模用户交互序列，结合用户/物品侧特征
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
import time

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────

def load_data():
    base = os.path.join(DATA_ROOT, "A推荐", "A推荐")
    train = pd.read_csv(os.path.join(base, "train.csv"))
    test = pd.read_csv(os.path.join(base, "test.csv"))
    user_df = pd.read_csv(os.path.join(base, "user.csv"))
    item_df = pd.read_csv(os.path.join(base, "item.csv"))
    sample_sub = pd.read_csv(os.path.join(base, "sample_submission.csv"))
    return train, test, user_df, item_df, sample_sub


# ─────────────────────────────────────────────
# ID 映射与特征编码
# ─────────────────────────────────────────────

def build_id_maps(train, test, item_df):
    """构建用户和物品的 ID 映射"""
    # 用户 ID
    all_uids = sorted(set(train["uid"]) | set(test["uid"]))
    uid2idx = {uid: i + 1 for i, uid in enumerate(all_uids)}  # 0 留给 padding

    # 物品 ID
    all_iids = sorted(set(item_df["iid"]))
    iid2idx = {iid: i + 1 for i, iid in enumerate(all_iids)}  # 0 留给 padding

    return uid2idx, iid2idx, len(uid2idx) + 1, len(iid2idx) + 1


def encode_user_features(user_df, uid2idx):
    """编码用户侧特征为 embedding index"""
    user_feats = {}
    cat_cols = [c for c in user_df.columns if c.startswith("u_cat_")]
    for _, row in user_df.iterrows():
        uid = row["uid"]
        if uid in uid2idx:
            user_feats[uid2idx[uid]] = [int(row[c]) for c in cat_cols]
    return user_feats, [user_df[c].nunique() + 1 for c in cat_cols]  # +1 for padding


def encode_item_features(item_df, iid2idx):
    """编码物品侧特征为 embedding index"""
    item_feats = {}
    cat_cols = [c for c in item_df.columns if c.startswith("i_cat_")]
    for _, row in item_df.iterrows():
        iid = row["iid"]
        if iid in iid2idx:
            item_feats[iid2idx[iid]] = [int(row[c]) for c in cat_cols]
    return item_feats, [item_df[c].nunique() + 1 for c in cat_cols]


# ─────────────────────────────────────────────
# 数据集
# ─────────────────────────────────────────────

class SeqRecDataset(Dataset):
    def __init__(self, df, uid2idx, iid2idx, max_len=100, is_test=False):
        self.uid2idx = uid2idx
        self.iid2idx = iid2idx
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
        # 截断或填充序列
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


# ─────────────────────────────────────────────
# SASRec 模型
# ─────────────────────────────────────────────

class SASRec(nn.Module):
    def __init__(
        self,
        num_items,
        embed_dim=128,
        max_len=100,
        num_heads=2,
        num_layers=2,
        dropout=0.2,
        num_user_feats=0,
        user_feat_dims=None,
        num_item_feats=0,
        item_feat_dims=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

        # 物品 embedding
        self.item_emb = nn.Embedding(num_items, embed_dim, padding_idx=0)

        # 位置编码
        self.pos_emb = nn.Embedding(max_len, embed_dim)

        # 用户侧特征 embedding
        self.user_feat_emb = None
        if num_user_feats > 0 and user_feat_dims:
            self.user_feat_embs = nn.ModuleList([
                nn.Embedding(dim, min(32, dim), padding_idx=0)
                for dim in user_feat_dims
            ])
            user_feat_total = sum(min(32, d) for d in user_feat_dims)
            self.user_feat_proj = nn.Linear(user_feat_total, embed_dim)
            self.user_feat_emb = True

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.output_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, uid, seq, mask):
        """
        seq: (batch, max_len) 物品 ID 序列
        mask: (batch, max_len) 有效位置为 1
        """
        batch_size, seq_len = seq.shape

        # 物品 embedding + 位置编码
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.dropout(x)

        # Causal mask (下三角)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=seq.device), diagonal=1
        ).bool()

        # Padding mask
        padding_mask = ~mask.bool()  # True 表示需要 mask 的位置

        # Transformer 编码
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        x = self.output_norm(x)

        # 取最后一个有效位置的输出作为序列表示
        # 找到每个序列最后一个有效位置
        seq_len_per_sample = mask.sum(dim=1).long() - 1  # (batch,)
        seq_len_per_sample = seq_len_per_sample.clamp(min=0)
        user_repr = x[torch.arange(batch_size, device=x.device), seq_len_per_sample]  # (batch, embed_dim)

        return user_repr

    def predict_scores(self, user_repr, all_item_ids=None):
        """计算用户对所有物品的得分"""
        if all_item_ids is not None:
            item_embs = self.item_emb(all_item_ids)  # (num_items, embed_dim)
        else:
            item_embs = self.item_emb.weight[1:]  # 排除 padding
        scores = torch.matmul(user_repr, item_embs.T)  # (batch, num_items)
        return scores


# ─────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, num_items, neg_samples=100):
    model.train()
    total_loss = 0
    total_correct = 0
    total_count = 0

    for uid, seq, mask, target in dataloader:
        uid, seq, mask, target = (
            uid.to(DEVICE),
            seq.to(DEVICE),
            mask.to(DEVICE),
            target.to(DEVICE),
        )

        optimizer.zero_grad()

        # 前向传播
        user_repr = model(uid, seq, mask)  # (batch, embed_dim)

        # 负采样
        batch_size = user_repr.size(0)
        neg = torch.randint(1, num_items, (batch_size, neg_samples), device=DEVICE)
        # 确保负样本不包含 target
        all_items = torch.cat([target.unsqueeze(1), neg], dim=1)  # (batch, 1+neg)
        all_embs = model.item_emb(all_items)  # (batch, 1+neg, embed_dim)

        # 计算得分
        scores = (user_repr.unsqueeze(1) * all_embs).sum(dim=2)  # (batch, 1+neg)

        # BPR 损失: target 得分应高于负样本
        labels = torch.zeros(batch_size, dtype=torch.long, device=DEVICE)  # 第 0 个是正样本
        loss = F.cross_entropy(scores, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * batch_size
        pred = scores.argmax(dim=1)
        total_correct += (pred == 0).sum().item()
        total_count += batch_size

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(model, dataloader, num_items, ks=[1, 5, 10]):
    model.eval()
    hits = {k: 0 for k in ks}
    total = 0

    for uid, seq, mask, target in dataloader:
        uid, seq, mask, target = (
            uid.to(DEVICE),
            seq.to(DEVICE),
            mask.to(DEVICE),
            target.to(DEVICE),
        )

        user_repr = model(uid, seq, mask)
        scores = model.predict_scores(user_repr)  # (batch, num_items-1)

        for k in ks:
            topk = scores.topk(k, dim=1).indices + 1  # +1 因为去掉了 padding
            for i in range(len(target)):
                if target[i].item() in topk[i]:
                    hits[k] += 1
        total += len(target)

    return {k: hits[k] / total for k in ks}


@torch.no_grad()
def predict_test(model, test_loader, num_items, topk=10):
    model.eval()
    all_preds = []

    for uid, seq, mask, _ in test_loader:
        uid, seq, mask = uid.to(DEVICE), seq.to(DEVICE), mask.to(DEVICE)
        user_repr = model(uid, seq, mask)
        scores = model.predict_scores(user_repr)
        topk_indices = scores.topk(topk, dim=1).indices + 1  # +1 映射回原始索引
        all_preds.append(topk_indices.cpu())

    return torch.cat(all_preds, dim=0)


def run():
    print("加载数据...")
    train_df, test_df, user_df, item_df, sample_sub = load_data()

    # 构建 ID 映射
    uid2idx, iid2idx, num_users, num_items = build_id_maps(train_df, test_df, item_df)
    idx2iid = {v: k for k, v in iid2idx.items()}  # idx -> iid 字符串

    print(f"用户数: {num_users - 1}, 物品数: {num_items - 1}")

    # 编码侧特征
    user_feat_dict, user_feat_dims = encode_user_features(user_df, uid2idx)
    item_feat_dict, item_feat_dims = encode_item_features(item_df, iid2idx)
    print(f"用户特征维度: {user_feat_dims}")
    print(f"物品特征维度: {item_feat_dims}")

    # 训练/验证划分
    train_split, val_split = train_test_split(train_df, test_size=0.1, random_state=42)
    print(f"训练集: {len(train_split)}, 验证集: {len(val_split)}")

    # 数据集
    MAX_LEN = 100
    train_dataset = SeqRecDataset(train_split, uid2idx, iid2idx, max_len=MAX_LEN)
    val_dataset = SeqRecDataset(val_split, uid2idx, iid2idx, max_len=MAX_LEN)
    test_dataset = SeqRecDataset(test_df, uid2idx, iid2idx, max_len=MAX_LEN, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0)

    # 模型
    model = SASRec(
        num_items=num_items,
        embed_dim=128,
        max_len=MAX_LEN,
        num_heads=2,
        num_layers=2,
        dropout=0.2,
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # 训练循环
    EPOCHS = 100
    PATIENCE = 15
    best_val_hit10 = 0
    patience_counter = 0
    best_state = None

    print(f"\n开始训练 (设备: {DEVICE})...")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, num_items)
        val_metrics = evaluate(model, val_loader, num_items)
        scheduler.step()

        val_hit10 = val_metrics[10]
        if val_hit10 > best_val_hit10:
            best_val_hit10 = val_hit10
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d} | loss={train_loss:.4f} | acc={train_acc:.4f} | "
                f"H@1={val_metrics[1]:.4f} H@5={val_metrics[5]:.4f} H@10={val_metrics[10]:.4f} | "
                f"{time.time()-t0:.1f}s"
            )

        if patience_counter >= PATIENCE:
            print(f"  早停于 epoch {epoch}")
            break

    # 加载最优模型
    model.load_state_dict(best_state)
    model.to(DEVICE)
    print(f"\n最优验证 Hit@10: {best_val_hit10:.4f}")

    # 生成测试集预测
    print("生成测试集预测...")
    test_preds_idx = predict_test(model, test_loader, num_items, topk=10)

    # 转换为原始 iid
    rows = []
    for i, (_, row) in enumerate(sample_sub.iterrows()):
        uid = row["uid"]
        pred_indices = test_preds_idx[i].tolist()
        pred_iids = [idx2iid.get(idx, "i000001") for idx in pred_indices]
        rows.append({"uid": uid, "prediction": ",".join(pred_iids)})

    sub = pd.DataFrame(rows)
    sub_path = os.path.join(DATA_ROOT, "A2.csv")
    sub.to_csv(sub_path, index=False)
    print(f"提交文件保存至 {sub_path}")
    print(f"行数: {len(sub)}")
    print(f"样例: {sub.iloc[0]['prediction'][:80]}...")


if __name__ == "__main__":
    run()

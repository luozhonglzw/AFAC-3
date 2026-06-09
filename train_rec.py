"""
产品推荐任务 - 训练脚本
SASRec 序列推荐模型，支持用户/物品特征增强
"""

import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# 数据加载与预处理
# ─────────────────────────────────────────────

def load_data():
    base = os.path.join(DATA_ROOT, "A推荐", "A推荐")
    train = pd.read_csv(os.path.join(base, "train.csv"))
    test = pd.read_csv(os.path.join(base, "test.csv"))
    user_df = pd.read_csv(os.path.join(base, "user.csv"))
    item_df = pd.read_csv(os.path.join(base, "item.csv"))
    sample_sub = pd.read_csv(os.path.join(base, "sample_submission.csv"))
    return train, test, user_df, item_df, sample_sub


def build_id_maps(train, test, item_df):
    all_uids = sorted(set(train["uid"]) | set(test["uid"]))
    uid2idx = {uid: i + 1 for i, uid in enumerate(all_uids)}
    all_iids = sorted(set(item_df["iid"]))
    iid2idx = {iid: i + 1 for i, iid in enumerate(all_iids)}
    return uid2idx, iid2idx, len(uid2idx) + 1, len(iid2idx) + 1


class RecDataset(Dataset):
    def __init__(self, df, uid2idx, iid2idx, max_len=100, is_test=False):
        self.max_len = max_len
        self.samples = []
        for _, row in df.iterrows():
            uid = uid2idx[row["uid"]]
            seq_raw = str(row["item_seq_raw"]).strip()
            if not seq_raw or seq_raw == "nan":
                seq = []
            else:
                seq = [iid2idx.get(x, 0) for x in seq_raw.split(",")]
            if not is_test:
                target = iid2idx.get(row["target_iid"], 0)
                self.samples.append((uid, seq, target))
            else:
                self.samples.append((uid, seq, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uid, seq, target = self.samples[idx]
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]
        padded = [0] * (self.max_len - len(seq)) + seq
        mask = [0] * (self.max_len - len(seq)) + [1] * len(seq)
        return (
            torch.tensor(uid, dtype=torch.long),
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float),
            torch.tensor(target, dtype=torch.long),
        )


# ─────────────────────────────────────────────
# SASRec 模型
# ─────────────────────────────────────────────

class SASRec(nn.Module):
    def __init__(self, num_items, embed_dim=128, max_len=100,
                 num_heads=2, num_layers=2, dropout=0.2):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, seq, mask):
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0)
        x = self.drop(self.item_emb(seq) + self.pos_emb(pos))
        causal = torch.triu(torch.ones(L, L, device=seq.device), diagonal=1).bool()
        pad_mask = ~mask.bool()
        x = self.encoder(x, mask=causal, src_key_padding_mask=pad_mask)
        x = self.norm(x)
        idx = mask.sum(1).long() - 1
        idx = idx.clamp(min=0)
        return x[torch.arange(B, device=x.device), idx]


# ─────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────

def train():
    print("加载数据...")
    train_df, test_df, user_df, item_df, sample_sub = load_data()
    uid2idx, iid2idx, num_users, num_items = build_id_maps(train_df, test_df, item_df)

    config = {
        "embed_dim": 128,
        "num_heads": 2,
        "num_layers": 2,
        "dropout": 0.2,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "max_len": 100,
        "batch_size": 256,
        "neg_samples": 100,
        "epochs": 100,
        "patience": 15,
    }
    print(f"配置: {json.dumps(config, indent=2)}")
    print(f"用户数: {num_users-1}, 物品数: {num_items-1}, 设备: {DEVICE}")

    # 划分
    trn, val = train_test_split(train_df, test_size=0.1, random_state=42)
    trn_ds = RecDataset(trn, uid2idx, iid2idx, config["max_len"])
    val_ds = RecDataset(val, uid2idx, iid2idx, config["max_len"])
    trn_loader = DataLoader(trn_ds, config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, 512, shuffle=False, num_workers=0)

    # 模型
    model = SASRec(num_items, config["embed_dim"], config["max_len"],
                   config["num_heads"], config["num_layers"], config["dropout"]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config["epochs"])

    best_hit10 = 0
    patience_cnt = 0
    best_state = None

    print("\n开始训练...")
    for ep in range(1, config["epochs"] + 1):
        t0 = time.time()

        # 训练
        model.train()
        total_loss = 0
        total_correct = 0
        total_n = 0
        for uid, seq, mask, target in trn_loader:
            uid, seq, mask, target = [x.to(DEVICE) for x in (uid, seq, mask, target)]
            opt.zero_grad()
            repr_ = model(seq, mask)
            B = repr_.size(0)
            neg = torch.randint(1, num_items, (B, config["neg_samples"]), device=DEVICE)
            items = torch.cat([target.unsqueeze(1), neg], 1)
            embs = model.item_emb(items)
            scores = (repr_.unsqueeze(1) * embs).sum(2)
            labels = torch.zeros(B, dtype=torch.long, device=DEVICE)
            loss = F.cross_entropy(scores, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item() * B
            total_correct += (scores.argmax(1) == 0).sum().item()
            total_n += B
        sched.step()

        # 验证
        model.eval()
        hits = {1: 0, 5: 0, 10: 0}
        val_n = 0
        with torch.no_grad():
            for uid, seq, mask, target in val_loader:
                uid, seq, mask, target = [x.to(DEVICE) for x in (uid, seq, mask, target)]
                repr_ = model(seq, mask)
                all_embs = model.item_emb.weight[1:]
                scores = torch.matmul(repr_, all_embs.T)
                for k in [1, 5, 10]:
                    topk = scores.topk(k, 1).indices + 1
                    for i in range(len(target)):
                        if target[i].item() in topk[i]:
                            hits[k] += 1
                val_n += len(target)

        h10 = hits[10] / val_n
        if h10 > best_hit10:
            best_hit10 = h10
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1

        if ep % 5 == 0 or ep == 1:
            print(f"  Ep {ep:3d} | loss={total_loss/total_n:.4f} | "
                  f"H@1={hits[1]/val_n:.4f} H@5={hits[5]/val_n:.4f} H@10={h10:.4f} | "
                  f"{time.time()-t0:.1f}s")

        if patience_cnt >= config["patience"]:
            print(f"  早停于 epoch {ep}")
            break

    print(f"\n最优 Hit@10: {best_hit10:.4f}")

    # 保存
    save_dir = os.path.join(DATA_ROOT, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        "model_state": best_state,
        "config": config,
        "uid2idx": uid2idx,
        "iid2idx": iid2idx,
        "num_items": num_items,
        "best_hit10": best_hit10,
    }, os.path.join(save_dir, "rec_best.pt"))
    print(f"模型已保存至 checkpoints/rec_best.pt")


if __name__ == "__main__":
    train()

"""
产品推荐任务 - 预测脚本
加载训练好的 SASRec 模型，生成 A2.csv 提交文件
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_rec import SASRec, RecDataset, load_data

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict():
    # 加载 checkpoint
    ckpt_path = os.path.join(DATA_ROOT, "checkpoints", "rec_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到模型文件: {ckpt_path}\n请先运行 train_rec.py")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    uid2idx = ckpt["uid2idx"]
    iid2idx = ckpt["iid2idx"]
    num_items = ckpt["num_items"]
    idx2iid = {v: k for k, v in iid2idx.items()}

    print(f"加载模型: Hit@10={ckpt['best_hit10']:.4f}")

    # 加载数据
    train_df, test_df, user_df, item_df, sample_sub = load_data()

    # 测试数据集
    test_ds = RecDataset(test_df, uid2idx, iid2idx, config["max_len"], is_test=True)
    test_loader = DataLoader(test_ds, 512, shuffle=False, num_workers=0)

    # 构建并加载模型
    model = SASRec(
        num_items, config["embed_dim"], config["max_len"],
        config["num_heads"], config["num_layers"], config["dropout"]
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 预测
    print("生成 Top-10 预测...")
    all_preds = []
    with torch.no_grad():
        for uid, seq, mask, _ in test_loader:
            uid, seq, mask = [x.to(DEVICE) for x in (uid, seq, mask)]
            repr_ = model(seq, mask)
            all_embs = model.item_emb.weight[1:]
            scores = torch.matmul(repr_, all_embs.T)
            topk = scores.topk(10, dim=1).indices + 1
            all_preds.append(topk.cpu())

    preds = torch.cat(all_preds, dim=0)

    # 转换为原始 iid 并保存
    rows = []
    for i, (_, row) in enumerate(sample_sub.iterrows()):
        uid = row["uid"]
        pred_iids = [idx2iid.get(idx.item(), "i000001") for idx in preds[i]]
        rows.append({"uid": uid, "prediction": ",".join(pred_iids)})

    sub = pd.DataFrame(rows)
    out_path = os.path.join(DATA_ROOT, "A2.csv")
    sub.to_csv(out_path, index=False)

    print(f"提交文件已保存: {out_path}")
    print(f"行数: {len(sub)}")
    print(f"样例: {sub.iloc[0]['prediction'][:80]}...")


if __name__ == "__main__":
    predict()

"""
产品分类任务 - 预测脚本
加载训练好的模型，生成 A1.csv 提交文件
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, diags
from sklearn.preprocessing import StandardScaler

# 复用训练脚本的模型定义和数据处理
from train_cls import GCN, MLP, load_data, preprocess_graph, to_sparse_tensor

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict():
    # 加载 checkpoint
    ckpt_path = os.path.join(DATA_ROOT, "checkpoints", "cls_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到模型文件: {ckpt_path}\n请先运行 train_cls.py")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    print(f"加载模型: val_acc={ckpt['best_val_acc']:.4f}")

    # 加载数据
    adj, features, labels, train_idx, test_idx = load_data()

    # 特征预处理 (与训练一致)
    if config["feature_norm"]:
        feat = StandardScaler().fit_transform(features.toarray())
    else:
        feat = features.toarray()
    feat_t = torch.from_numpy(feat.astype(np.float32)).to(DEVICE)

    # 图预处理 (与训练一致)
    adj_norm = preprocess_graph(adj, config["symmetrize"], config["norm_mode"])
    adj_t = to_sparse_tensor(adj_norm).to(DEVICE)

    # 构建并加载模型
    if config["model_type"] == "GCN":
        model = GCN(767, config["hidden_dim"], 10, config["num_layers"], config["dropout"]).to(DEVICE)
    else:
        model = MLP(767, config["hidden_dim"], 10, config["num_layers"], config["dropout"]).to(DEVICE)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 预测
    print("生成预测...")
    with torch.no_grad():
        out = model(feat_t, adj_t)
        test_pred = out[test_idx].argmax(dim=1).cpu().numpy()

    # 保存提交文件
    sample = pd.read_csv(os.path.join(DATA_ROOT, "A分类", "A分类", "sample_submission.csv"))
    sample["label"] = test_pred
    out_path = os.path.join(DATA_ROOT, "A1.csv")
    sample.to_csv(out_path, index=False)

    print(f"提交文件已保存: {out_path}")
    print(f"行数: {len(sample)}")
    print(f"预测分布: {np.bincount(test_pred, minlength=10).tolist()}")


if __name__ == "__main__":
    predict()

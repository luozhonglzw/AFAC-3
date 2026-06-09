"""
产品分类任务 - 预测脚本 (V2: GCN + Label Propagation 集成)
手动实现稀疏图标签传播，与 GCN 加权融合
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, diags, eye as speye
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from train_cls import GCN, MLP, load_data, preprocess_graph, to_sparse_tensor

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# 稀疏图标签传播
# ─────────────────────────────────────────────

def run_label_propagation(adj, labels, train_idx, test_idx, num_iters=30, alpha=0.2):
    """
    稀疏图标签传播 (Graph Label Propagation)
    使用稀疏矩阵迭代传播，避免 dense 矩阵 OOM

    算法:
    1. 构建行归一化转移矩阵 T = D^{-1} A
    2. 初始化标签分布 Y (训练节点 one-hot, 测试节点均匀)
    3. 迭代: Y_new = (1-alpha) * T @ Y_old + alpha * Y_init
    4. 保持训练节点标签不变 (clamp)
    """
    print(f"\n[LP] 运行稀疏图标签传播 (iters={num_iters}, alpha={alpha})...")

    N = adj.shape[0]
    num_classes = 10

    # 构建对称邻接矩阵 + 自环
    adj_sym = adj + adj.T
    adj_sym.data = np.ones_like(adj_sym.data, dtype=np.float32)
    adj_sym = adj_sym + speye(N, format="csr", dtype=np.float32)

    # 行归一化: T = D^{-1} A
    deg = np.array(adj_sym.sum(axis=1)).flatten()
    deg_inv = np.power(deg, -1.0)
    deg_inv[np.isinf(deg_inv)] = 0.0
    T = diags(deg_inv) @ adj_sym

    t0 = time.time()

    # 初始化标签分布 Y
    Y = np.ones((N, num_classes), dtype=np.float64) / num_classes
    Y_init = Y.copy()

    # 训练节点: one-hot 编码
    train_set = set(train_idx.tolist())
    for idx in train_idx:
        Y[idx] = 0
        Y[idx, labels[idx]] = 1.0
    Y_init[:] = Y

    # 迭代传播
    for it in range(num_iters):
        Y_new = (1 - alpha) * (T @ Y) + alpha * Y_init
        # Clamp 训练节点标签
        for idx in train_idx:
            Y_new[idx] = 0
            Y_new[idx, labels[idx]] = 1.0
        diff = np.abs(Y_new - Y).max()
        Y = Y_new
        if (it + 1) % 10 == 0:
            print(f"  iter {it+1}: max_diff={diff:.6f}")
        if diff < 1e-6:
            print(f"  收敛于 iter {it+1}")
            break

    elapsed = time.time() - t0

    # 归一化为概率分布
    row_sums = Y.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    lp_probs = Y / row_sums

    # 验证集评估 (无泄露: 只用 trn_sub 传播, 在 val_sub 上评估)
    trn_sub, val_sub = train_test_split(
        train_idx, test_size=0.2, random_state=42, stratify=labels[train_idx]
    )

    # 重新用 trn_sub 做一次传播 (不含 val_sub, 避免数据泄露)
    Y_val = np.ones((N, num_classes), dtype=np.float64) / num_classes
    Y_val_init = Y_val.copy()
    for idx in trn_sub:
        Y_val[idx] = 0
        Y_val[idx, labels[idx]] = 1.0
    Y_val_init[:] = Y_val
    for it in range(num_iters):
        Y_val_new = (1 - alpha) * (T @ Y_val) + alpha * Y_val_init
        for idx in trn_sub:
            Y_val_new[idx] = 0
            Y_val_new[idx, labels[idx]] = 1.0
        Y_val = Y_val_new
    row_sums_val = Y_val.sum(axis=1, keepdims=True)
    row_sums_val[row_sums_val == 0] = 1
    lp_val_probs = Y_val / row_sums_val
    val_pred = lp_val_probs[val_sub].argmax(axis=1)
    val_acc = (val_pred == labels[val_sub]).mean()

    print(f"[LP] 完成 | 耗时: {elapsed:.1f}s | 验证准确率: {val_acc:.4f} (无泄露)")

    return lp_probs, val_acc


# ─────────────────────────────────────────────
# GCN 预测器
# ─────────────────────────────────────────────

def run_gcn_predict(config, adj, features, labels, train_idx, test_idx):
    """GCN 模型预测，返回全图概率分布"""
    print(f"\n[GCN] 加载模型并预测...")

    if config["feature_norm"]:
        feat = StandardScaler().fit_transform(features.toarray())
    else:
        feat = features.toarray()
    feat_t = torch.from_numpy(feat.astype(np.float32)).to(DEVICE)

    adj_norm = preprocess_graph(adj, config["symmetrize"], config["norm_mode"])
    adj_t = to_sparse_tensor(adj_norm).to(DEVICE)

    ckpt_path = os.path.join(DATA_ROOT, "checkpoints", "cls_best.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    if config["model_type"] == "GCN":
        model = GCN(767, config["hidden_dim"], 10, config["num_layers"], config["dropout"]).to(DEVICE)
    else:
        model = MLP(767, config["hidden_dim"], 10, config["num_layers"], config["dropout"]).to(DEVICE)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        out = model(feat_t, adj_t)
        gcn_all_probs = F.softmax(out, dim=1).cpu().numpy()

    # 验证集评估
    trn_sub, val_sub = train_test_split(
        train_idx, test_size=0.2, random_state=42, stratify=labels[train_idx]
    )
    val_pred = gcn_all_probs[val_sub].argmax(axis=1)
    val_acc = (val_pred == labels[val_sub]).mean()

    print(f"[GCN] 验证准确率: {val_acc:.4f}")

    return gcn_all_probs, val_acc


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def predict():
    print("=" * 60)
    print("  GCN + Label Propagation 集成预测")
    print("=" * 60)

    adj, features, labels, train_idx, test_idx = load_data()

    # 加载 GCN 配置
    ckpt_path = os.path.join(DATA_ROOT, "checkpoints", "cls_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到模型文件: {ckpt_path}\n请先运行 train_cls.py")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    print(f"GCN 配置: val_acc={ckpt['best_val_acc']:.4f}")

    # 1. GCN 全图概率
    gcn_all_probs, gcn_val_acc = run_gcn_predict(config, adj, features, labels, train_idx, test_idx)

    # 2. LP 全图概率
    lp_all_probs, lp_val_acc = run_label_propagation(adj, labels, train_idx, test_idx)

    # 3. 验证集策略比较 (使用无泄露的 LP 概率做验证)
    trn_sub, val_sub = train_test_split(
        train_idx, test_size=0.2, random_state=42, stratify=labels[train_idx]
    )

    # 用无泄露的 LP 概率做验证比较
    # lp_val_probs 已在 run_label_propagation 中计算 (只用 trn_sub 传播)
    # 重新获取无泄露 LP 概率用于验证
    N = adj.shape[0]
    adj_sym_val = adj + adj.T
    adj_sym_val.data = np.ones_like(adj_sym_val.data, dtype=np.float32)
    adj_sym_val = adj_sym_val + speye(N, format="csr", dtype=np.float32)
    deg_val = np.array(adj_sym_val.sum(axis=1)).flatten()
    deg_inv_val = np.power(deg_val, -1.0)
    deg_inv_val[np.isinf(deg_inv_val)] = 0.0
    T_val = diags(deg_inv_val) @ adj_sym_val
    Y_val_lp = np.ones((N, 10), dtype=np.float64) / 10
    Y_val_lp_init = Y_val_lp.copy()
    for idx in trn_sub:
        Y_val_lp[idx] = 0
        Y_val_lp[idx, labels[idx]] = 1.0
    Y_val_lp_init[:] = Y_val_lp
    for _ in range(30):
        Y_val_new = 0.8 * (T_val @ Y_val_lp) + 0.2 * Y_val_lp_init
        for idx in trn_sub:
            Y_val_new[idx] = 0
            Y_val_new[idx, labels[idx]] = 1.0
        Y_val_lp = Y_val_new
    row_sums_v = Y_val_lp.sum(axis=1, keepdims=True)
    row_sums_v[row_sums_v == 0] = 1
    lp_val_probs_no_leak = Y_val_lp / row_sums_v

    strategies = {
        "gcn_only": lambda g, l: g,
        "lp_only": lambda g, l: l,
        "average": lambda g, l: 0.5 * g + 0.5 * l,
        "weighted": lambda g, l: (gcn_val_acc / (gcn_val_acc + lp_val_acc)) * g
                                  + (lp_val_acc / (gcn_val_acc + lp_val_acc)) * l,
    }

    print(f"\n[验证集策略比较]")
    best_strategy = None
    best_val_acc = 0
    for name, fn in strategies.items():
        # 验证时用无泄露的 LP 概率
        probs = fn(gcn_all_probs, lp_val_probs_no_leak)
        val_pred = probs[val_sub].argmax(axis=1)
        vac = (val_pred == labels[val_sub]).mean()
        print(f"  {name:12s}: {vac:.4f}")
        if vac > best_val_acc:
            best_val_acc = vac
            best_strategy = name

    print(f"\n[最优策略] {best_strategy} (val_acc={best_val_acc:.4f})")

    # 4. 生成测试预测 (最终预测用全 train_idx 传播的 LP 概率, 这是正确的)
    fn = strategies[best_strategy]
    final_probs = fn(gcn_all_probs, lp_all_probs)
    test_pred = final_probs[test_idx].argmax(axis=1)

    # 5. 保存
    sample = pd.read_csv(os.path.join(DATA_ROOT, "A分类", "A分类", "sample_submission.csv"))
    sample["label"] = test_pred
    out_path = os.path.join(DATA_ROOT, "A1.csv")
    sample.to_csv(out_path, index=False)

    print(f"\n提交文件已保存: {out_path}")
    print(f"行数: {len(sample)}")
    print(f"预测分布: {np.bincount(test_pred, minlength=10).tolist()}")

    return best_val_acc


if __name__ == "__main__":
    predict()

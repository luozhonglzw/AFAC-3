"""Optimized recommendation: KMeans cold-start + warm strategy for short sequences"""

import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_data():
    base = os.path.join(DATA_ROOT, "A推荐", "A推荐")
    train = pd.read_csv(os.path.join(base, "train.csv"))
    test = pd.read_csv(os.path.join(base, "test.csv"))
    user_df = pd.read_csv(os.path.join(base, "user.csv"))
    item_df = pd.read_csv(os.path.join(base, "item.csv"))
    sample_sub = pd.read_csv(os.path.join(base, "sample_submission.csv"))
    return train, test, user_df, item_df, sample_sub


def seq_len(s):
    if pd.isna(s) or str(s).strip() == "" or str(s).strip() == "nan":
        return 0
    return len(str(s).split(","))


def ndcg_at_k(ranked_list, target_item, k=10):
    for i, item in enumerate(ranked_list[:k]):
        if item == target_item:
            return 1.0 / np.log2(i + 2)
    return 0.0


def build_cluster_recs(user_df, train_df, k=20):
    """Build KMeans cluster-based recommendations."""
    feat_cols = [c for c in user_df.columns if c.startswith("u_cat_")]
    X = user_df[feat_cols].values.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    uid_to_cluster = {}
    for i in range(len(user_df)):
        uid_to_cluster[user_df.iloc[i]["uid"]] = clusters[i]

    cluster_targets = defaultdict(Counter)
    for _, row in train_df.iterrows():
        uid = row["uid"]
        target = row["target_iid"]
        if uid in uid_to_cluster:
            cluster_targets[uid_to_cluster[uid]][target] += 1

    cluster_recs = {}
    for c in range(k):
        cluster_recs[c] = [iid for iid, _ in cluster_targets[c].most_common(10)]

    return uid_to_cluster, cluster_recs, kmeans, scaler


def run():
    print("Loading data...")
    train_df, test_df, user_df, item_df, sample_sub = load_data()

    print("Building statistics...")
    item_counts = Counter()
    for seq in train_df["item_seq_raw"].dropna():
        for item_id in str(seq).split(","):
            item_counts[item_id] += 1

    target_dist = Counter(train_df["target_iid"])

    transitions = defaultdict(Counter)
    for _, row in train_df.iterrows():
        seq = str(row["item_seq_raw"]).strip()
        if not seq or seq == "nan":
            continue
        items = seq.split(",")
        for i in range(len(items) - 1):
            transitions[items[i]][items[i + 1]] += 1

    cooccur = defaultdict(Counter)
    for _, row in train_df.iterrows():
        seq = str(row["item_seq_raw"]).strip()
        if not seq or seq == "nan":
            continue
        items = list(set(seq.split(",")))
        for i in range(len(items)):
            for j in range(len(items)):
                if i != j:
                    cooccur[items[i]][items[j]] += 1

    last_to_target = defaultdict(Counter)
    for _, row in train_df.iterrows():
        target = row["target_iid"]
        seq_dedup = str(row["item_seq_dedup"]).strip()
        if not seq_dedup or seq_dedup == "nan":
            continue
        items = seq_dedup.split(",")
        last_to_target[items[-1]][target] += 1

    # Build cluster recommendations
    print("Building KMeans clusters (K=20)...")
    uid_to_cluster, cluster_recs, kmeans, scaler = build_cluster_recs(user_df, train_df, k=20)

    # Item similarity from categories
    cat_features = item_df[["i_cat_01", "i_cat_02", "i_cat_03"]].values
    iid_list = item_df["iid"].tolist()
    item_sim = defaultdict(dict)
    for i in range(len(iid_list)):
        for j in range(i + 1, len(iid_list)):
            shared = np.sum(cat_features[i] == cat_features[j])
            if shared > 0:
                sim = shared / 3.0
                item_sim[iid_list[i]][iid_list[j]] = sim
                item_sim[iid_list[j]][iid_list[i]] = sim

    def predict_warm(seq_raw, seq_dedup, topk=10):
        scores = defaultdict(float)
        items_raw = str(seq_raw).strip().split(",") if pd.notna(seq_raw) and str(seq_raw).strip() else []
        items_dedup = str(seq_dedup).strip().split(",") if pd.notna(seq_dedup) and str(seq_dedup).strip() else []
        raw_set = set(items_raw)
        for iid, cnt in item_counts.items():
            scores[iid] += 0.001 * cnt
        l2t_items = set()
        if items_dedup:
            last_item = items_dedup[-1]
            if last_item in transitions:
                for next_item, cnt in transitions[last_item].items():
                    scores[next_item] += 0.18 * cnt
            if last_item in last_to_target:
                for target, cnt in last_to_target[last_item].items():
                    scores[target] += 2000.0 * cnt
                    l2t_items.add(target)
        recent = items_dedup[-3:] if len(items_dedup) >= 3 else items_dedup
        for item in recent:
            if item in cooccur:
                for related, cnt in cooccur[item].items():
                    scores[related] += 0.18 * cnt
        user_freq = Counter(items_raw)
        for iid, cnt in user_freq.items():
            scores[iid] += 1000.0 * cnt
        for item in l2t_items:
            mult = 50.0
            if item in raw_set:
                mult *= 4.5
            scores[item] *= mult
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [iid for iid, _ in ranked[:topk]]

    def predict_cold(uid, topk=10):
        """Cold-start: cluster recommendation + popular fallback."""
        cluster_id = uid_to_cluster.get(uid, 0)
        rec = list(cluster_recs.get(cluster_id, []))
        if len(rec) < topk:
            for iid, _ in target_dist.most_common(topk):
                if iid not in rec:
                    rec.append(iid)
                if len(rec) >= topk:
                    break
        return rec[:topk]

    def predict_mixed(seq_raw, seq_dedup, uid, topk=10):
        """Mixed strategy based on sequence length."""
        sl = seq_len(seq_raw)

        if sl == 0:
            return predict_cold(uid, topk)

        # sl >= 3: pure warm (best NDCG)
        if sl >= 3:
            return predict_warm(seq_raw, seq_dedup, topk)

        # sl 1-2: mix warm with cluster
        warm_preds = predict_warm(seq_raw, seq_dedup, topk)
        cluster_rec = predict_cold(uid, topk)

        if sl == 1:
            warm_count, cluster_count = 6, 4
        else:  # sl == 2
            warm_count, cluster_count = 7, 3

        warm_top = warm_preds[:warm_count]
        cluster_top = [x for x in cluster_rec if x not in warm_top][:cluster_count]
        preds = warm_top + cluster_top

        if len(preds) < topk:
            for iid, _ in target_dist.most_common(topk * 2):
                if iid not in preds:
                    preds.append(iid)
                if len(preds) >= topk:
                    break
        return preds[:topk]

    # Evaluate on validation
    print("\n=== Validation ===")
    for sl in [1, 2, 3]:
        val_subset = train_df[train_df["item_seq_raw"].apply(seq_len) == sl]
        if len(val_subset) == 0:
            continue
        val_subset = val_subset.tail(min(1000, len(val_subset)))
        ndcg_scores = []
        hit_scores = []
        for _, row in val_subset.iterrows():
            preds = predict_mixed(row["item_seq_raw"], row["item_seq_dedup"], row["uid"])
            ndcg_scores.append(ndcg_at_k(preds, row["target_iid"]))
            hit_scores.append(1.0 if row["target_iid"] in preds else 0.0)
        print(f"  sl={sl}: NDCG={np.mean(ndcg_scores):.4f}, Hit={np.mean(hit_scores):.4f} (n={len(val_subset)})")

    # Test-like distribution validation
    print("\n=== Test-like Distribution ===")
    np.random.seed(42)
    sl1 = train_df[train_df["item_seq_raw"].apply(seq_len) == 1]
    sl2 = train_df[train_df["item_seq_raw"].apply(seq_len) == 2]
    sl3 = train_df[train_df["item_seq_raw"].apply(seq_len) == 3]
    sl4plus = train_df[train_df["item_seq_raw"].apply(seq_len) >= 4]

    val_parts = []
    if len(sl1) > 0:
        val_parts.append(sl1.sample(min(300, len(sl1))))
    if len(sl2) > 0:
        val_parts.append(sl2.sample(min(100, len(sl2))))
    if len(sl3) > 0:
        val_parts.append(sl3.sample(min(1300, len(sl3))))
    if len(sl4plus) > 0:
        val_parts.append(sl4plus.sample(min(300, len(sl4plus))))

    if val_parts:
        val_all = pd.concat(val_parts)
        ndcg_scores = []
        hit_scores = []
        for _, row in val_all.iterrows():
            preds = predict_mixed(row["item_seq_raw"], row["item_seq_dedup"], row["uid"])
            ndcg_scores.append(ndcg_at_k(preds, row["target_iid"]))
            hit_scores.append(1.0 if row["target_iid"] in preds else 0.0)
        print(f"  Overall: NDCG={np.mean(ndcg_scores):.4f}, Hit={np.mean(hit_scores):.4f} (n={len(val_all)})")

    # Generate test predictions
    print("\n=== Generating Test Predictions ===")
    predictions = {}
    cold_count = 0
    warm_count = 0
    for _, row in test_df.iterrows():
        uid = row["uid"]
        sl = seq_len(row["item_seq_raw"])
        preds = predict_mixed(row["item_seq_raw"], row["item_seq_dedup"], uid)
        predictions[uid] = preds
        if sl == 0:
            cold_count += 1
        else:
            warm_count += 1

    print(f"  Cold: {cold_count}, Warm: {warm_count}")

    rows = []
    for _, row in sample_sub.iterrows():
        uid = row["uid"]
        preds = predictions.get(uid, list(item_counts.keys())[:10])
        rows.append({"uid": uid, "prediction": ",".join(preds)})

    sub = pd.DataFrame(rows)
    out_path = os.path.join(DATA_ROOT, "A2.csv")
    sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    run()

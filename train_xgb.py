"""XGBoost ranking model for recommendation"""
import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import xgboost as xgb

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


def build_item_features(item_df):
    """Build item feature lookup."""
    item_feats = {}
    for _, row in item_df.iterrows():
        item_feats[row["iid"]] = {
            "i_cat_01": row["i_cat_01"],
            "i_cat_02": row["i_cat_02"],
            "i_cat_03": row["i_cat_03"],
            "i_bucket_01": row["i_bucket_01"],
        }
    return item_feats


def build_user_features(user_df):
    """Build user feature lookup."""
    user_feats = {}
    for _, row in user_df.iterrows():
        feats = {}
        for c in user_df.columns:
            if c.startswith("u_cat_"):
                feats[c] = row[c]
        user_feats[row["uid"]] = feats
    return user_feats


def build_training_data(train_df, item_feats, user_feats, item_counts, transitions,
                        last_to_target, item_cat_map, iid_list, neg_per_pos=9):
    """Build positive and negative samples for training."""
    all_items = list(item_counts.keys())
    X_rows = []
    y_labels = []

    for _, row in train_df.iterrows():
        uid = row["uid"]
        target = row["target_iid"]
        seq_raw = str(row["item_seq_raw"]).strip()
        seq_dedup = str(row["item_seq_dedup"]).strip()

        if not seq_raw or seq_raw == "nan":
            continue

        items_raw = seq_raw.split(",")
        items_dedup = seq_dedup.split(",") if seq_dedup and seq_dedup != "nan" else []
        raw_set = set(items_raw)
        last_item = items_dedup[-1] if items_dedup else None

        # User features
        u_feats = user_feats.get(uid, {})
        hist_len = len(items_raw)
        hist_unique = len(set(items_raw))

        # Positive sample
        feat = extract_features(uid, target, u_feats, item_feats, item_counts,
                                transitions, last_to_target, item_cat_map,
                                items_raw, items_dedup, raw_set, last_item, hist_len, hist_unique)
        X_rows.append(feat)
        y_labels.append(1)

        # Negative samples
        neg_items = set()
        attempts = 0
        while len(neg_items) < neg_per_pos and attempts < neg_per_pos * 3:
            neg_item = all_items[np.random.randint(len(all_items))]
            if neg_item != target and neg_item not in raw_set:
                neg_items.add(neg_item)
            attempts += 1

        for neg_item in neg_items:
            feat = extract_features(uid, neg_item, u_feats, item_feats, item_counts,
                                    transitions, last_to_target, item_cat_map,
                                    items_raw, items_dedup, raw_set, last_item, hist_len, hist_unique)
            X_rows.append(feat)
            y_labels.append(0)

    return np.array(X_rows), np.array(y_labels)


def extract_features(uid, iid, u_feats, item_feats, item_counts, transitions,
                     last_to_target, item_cat_map, items_raw, items_dedup,
                     raw_set, last_item, hist_len, hist_unique):
    """Extract features for a user-item pair."""
    i_feats = item_feats.get(iid, {"i_cat_01": 0, "i_cat_02": 0, "i_cat_03": 0, "i_bucket_01": 0})

    # User features
    u_cat_01 = u_feats.get("u_cat_01", 0)
    u_cat_02 = u_feats.get("u_cat_02", 0)
    u_cat_03 = u_feats.get("u_cat_03", 0)
    u_cat_04 = u_feats.get("u_cat_04", 0)
    u_cat_05 = u_feats.get("u_cat_05", 0)
    u_cat_06 = u_feats.get("u_cat_06", 0)
    u_cat_07 = u_feats.get("u_cat_07", 0)
    u_cat_08 = u_feats.get("u_cat_08", 0)

    # Item features
    i_cat_01 = i_feats["i_cat_01"]
    i_cat_02 = i_feats["i_cat_02"]
    i_cat_03 = i_feats["i_cat_03"]
    i_bucket_01 = i_feats["i_bucket_01"]

    # Popularity
    item_pop = item_counts.get(iid, 0)
    log_pop = np.log1p(item_pop)

    # Is in history
    in_history = 1 if iid in raw_set else 0

    # Is last item
    is_last = 1 if iid == last_item else 0

    # L2T signal
    l2t_score = 0
    if last_item and last_item in last_to_target:
        l2t_score = last_to_target[last_item].get(iid, 0)

    # Transition probability
    trans_score = 0
    if last_item and last_item in transitions:
        trans_score = transitions[last_item].get(iid, 0)

    # Category match with last item
    last_cat = item_cat_map.get(last_item, (0, 0, 0)) if last_item else (0, 0, 0)
    cand_cat = (i_cat_01, i_cat_02, i_cat_03)
    cat_match = sum(1 for a, b in zip(cand_cat, last_cat) if a == b) / 3.0

    # Category match with history
    hist_cat_match = 0
    if items_dedup:
        for item in items_dedup[-3:]:
            item_cat = item_cat_map.get(item, (0, 0, 0))
            if item_cat == cand_cat:
                hist_cat_match = 1
                break

    # Co-occurrence with recent items
    cooccur_score = 0
    for item in items_dedup[-3:]:
        if item in transitions:
            cooccur_score += transitions[item].get(iid, 0)

    return [
        hist_len, hist_unique,
        u_cat_01, u_cat_02, u_cat_03, u_cat_04, u_cat_05, u_cat_06, u_cat_07, u_cat_08,
        i_cat_01, i_cat_02, i_cat_03, i_bucket_01,
        log_pop, in_history, is_last,
        l2t_score, trans_score, cat_match, hist_cat_match, cooccur_score,
    ]


def run():
    print("Loading data...")
    train_df, test_df, user_df, item_df, sample_sub = load_data()

    print("Building features...")
    item_feats = build_item_features(item_df)
    user_feats = build_user_features(user_df)

    item_counts = Counter()
    for seq in train_df["item_seq_raw"].dropna():
        for item_id in str(seq).split(","):
            item_counts[item_id] += 1

    transitions = defaultdict(Counter)
    for _, row in train_df.iterrows():
        seq = str(row["item_seq_raw"]).strip()
        if not seq or seq == "nan":
            continue
        items = seq.split(",")
        for i in range(len(items) - 1):
            transitions[items[i]][items[i + 1]] += 1

    last_to_target = defaultdict(Counter)
    for _, row in train_df.iterrows():
        target = row["target_iid"]
        seq_dedup = str(row["item_seq_dedup"]).strip()
        if not seq_dedup or seq_dedup == "nan":
            continue
        items = seq_dedup.split(",")
        last_to_target[items[-1]][target] += 1

    item_cat_map = {}
    for _, row in item_df.iterrows():
        item_cat_map[row["iid"]] = (row["i_cat_01"], row["i_cat_02"], row["i_cat_03"])

    iid_list = item_df["iid"].tolist()
    target_dist = Counter(train_df["target_iid"])

    # Build training data
    print("Building training samples...")
    np.random.seed(42)
    X_train, y_train = build_training_data(
        train_df.head(30000), item_feats, user_feats, item_counts,
        transitions, last_to_target, item_cat_map, iid_list, neg_per_pos=9
    )
    print(f"  Training samples: {len(X_train)} (pos={sum(y_train)}, neg={len(y_train)-sum(y_train)})")

    # Train XGBoost
    print("Training XGBoost...")
    model = xgb.XGBClassifier(
        max_depth=6,
        n_estimators=100,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    # Feature importance
    feature_names = [
        "hist_len", "hist_unique",
        "u_cat_01", "u_cat_02", "u_cat_03", "u_cat_04", "u_cat_05", "u_cat_06", "u_cat_07", "u_cat_08",
        "i_cat_01", "i_cat_02", "i_cat_03", "i_bucket_01",
        "log_pop", "in_history", "is_last",
        "l2t_score", "trans_score", "cat_match", "hist_cat_match", "cooccur_score",
    ]
    importances = model.feature_importances_
    print("\nFeature importance:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1])[:10]:
        print(f"  {name}: {imp:.4f}")

    # Validate on short sequences
    print("\n=== Validation ===")
    for sl in [1, 2, 3]:
        val_sub = train_df.tail(5000)
        val_sub = val_sub[val_sub["item_seq_raw"].apply(seq_len) == sl]
        if len(val_sub) == 0:
            continue
        val_sub = val_sub.tail(min(500, len(val_sub)))

        ndcg_scores = []
        hit_scores = []
        for _, row in val_sub.iterrows():
            preds = predict_xgb(row, model, item_feats, user_feats, item_counts,
                                transitions, last_to_target, item_cat_map, iid_list, target_dist)
            ndcg_scores.append(ndcg_at_k(preds, row["target_iid"]))
            hit_scores.append(1.0 if row["target_iid"] in preds else 0.0)
        print(f"  sl={sl}: NDCG={np.mean(ndcg_scores):.4f}, Hit={np.mean(hit_scores):.4f} (n={len(val_sub)})")

    # Generate test predictions
    print("\n=== Generating Test Predictions ===")
    predictions = {}
    for _, row in test_df.iterrows():
        uid = row["uid"]
        preds = predict_xgb(row, model, item_feats, user_feats, item_counts,
                            transitions, last_to_target, item_cat_map, iid_list, target_dist)
        predictions[uid] = preds

    rows = []
    for _, row in sample_sub.iterrows():
        uid = row["uid"]
        preds = predictions.get(uid, list(item_counts.keys())[:10])
        rows.append({"uid": uid, "prediction": ",".join(preds)})

    sub = pd.DataFrame(rows)
    out_path = os.path.join(DATA_ROOT, "A2.csv")
    sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def predict_xgb(row, model, item_feats, user_feats, item_counts, transitions,
                last_to_target, item_cat_map, iid_list, target_dist, topk=10):
    """Predict using XGBoost model."""
    uid = row["uid"]
    seq_raw = str(row["item_seq_raw"]).strip()
    seq_dedup = str(row["item_seq_dedup"]).strip()

    if not seq_raw or seq_raw == "nan":
        return [iid for iid, _ in target_dist.most_common(topk)]

    items_raw = seq_raw.split(",")
    items_dedup = seq_dedup.split(",") if seq_dedup and seq_dedup != "nan" else []
    raw_set = set(items_raw)
    last_item = items_dedup[-1] if items_dedup else None
    u_feats = user_feats.get(uid, {})
    hist_len = len(items_raw)
    hist_unique = len(set(items_raw))

    # Generate candidates: top 50 from warm strategy + popular items
    candidates = set()
    # Add popular items
    for iid, _ in target_dist.most_common(20):
        candidates.add(iid)
    # Add l2t items
    if last_item and last_item in last_to_target:
        for iid, _ in last_to_target[last_item].most_common(20):
            candidates.add(iid)
    # Add transition items
    if last_item and last_item in transitions:
        for iid, _ in transitions[last_item].most_common(10):
            candidates.add(iid)
    # Add user history items
    for item in items_raw:
        candidates.add(item)

    # Score candidates
    X_cand = []
    cand_list = []
    for iid in candidates:
        feat = extract_features(uid, iid, u_feats, item_feats, item_counts,
                                transitions, last_to_target, item_cat_map,
                                items_raw, items_dedup, raw_set, last_item, hist_len, hist_unique)
        X_cand.append(feat)
        cand_list.append(iid)

    if not X_cand:
        return [iid for iid, _ in target_dist.most_common(topk)]

    X_cand = np.array(X_cand)
    probs = model.predict_proba(X_cand)[:, 1]
    ranked_indices = np.argsort(probs)[::-1]
    preds = [cand_list[i] for i in ranked_indices[:topk]]

    # Fill with popular if needed
    if len(preds) < topk:
        for iid, _ in target_dist.most_common(topk * 2):
            if iid not in preds:
                preds.append(iid)
            if len(preds) >= topk:
                break

    return preds[:topk]


if __name__ == "__main__":
    run()

"""Test deeper XGBRanker trees"""
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import xgboost as xgb
import sys
sys.path.insert(0, '.')
from train_final import load_data, extract_features, build_item_features, build_user_features, seq_len

train_df, test_df, user_df, item_df, sample_sub = load_data()
item_feats = build_item_features(item_df)
user_feats = build_user_features(user_df)

item_counts = Counter()
for seq in train_df['item_seq_raw'].dropna():
    for item in str(seq).split(','):
        item_counts[item] += 1

transitions = defaultdict(Counter)
pair_transitions = defaultdict(Counter)
for _, row in train_df.iterrows():
    seq = str(row['item_seq_raw']).strip()
    if not seq or seq == 'nan':
        continue
    items = seq.split(',')
    for i in range(len(items) - 1):
        transitions[items[i]][items[i+1]] += 1
    for i in range(len(items) - 2):
        pair_transitions[(items[i], items[i+1])][items[i+2]] += 1

last_to_target = defaultdict(Counter)
last2_to_target = defaultdict(Counter)
for _, row in train_df.iterrows():
    target = row['target_iid']
    seq_dedup = str(row['item_seq_dedup']).strip()
    if seq_dedup and seq_dedup != 'nan':
        items = seq_dedup.split(',')
        last_to_target[items[-1]][target] += 1
        if len(items) >= 2:
            last2_to_target[(items[-2], items[-1])][target] += 1

item_cat_map = {}
for _, row in item_df.iterrows():
    item_cat_map[row['iid']] = (row['i_cat_01'], row['i_cat_02'], row['i_cat_03'])

target_dist = Counter(train_df['target_iid'])

def ndcg_at_k(ranked_list, target_item, k=10):
    for i, item in enumerate(ranked_list[:k]):
        if item == target_item:
            return 1.0 / np.log2(i + 2)
    return 0.0

np.random.seed(42)
all_items = list(item_counts.keys())
X_rows = []
y_labels = []
group_sizes = []
neg_per_pos = 19

for _, row in train_df.iterrows():
    uid = row['uid']
    target = row['target_iid']
    seq_raw = str(row['item_seq_raw']).strip()
    seq_dedup = str(row['item_seq_dedup']).strip()
    if not seq_raw or seq_raw == 'nan':
        continue
    items_raw = seq_raw.split(',')
    items_dedup = seq_dedup.split(',') if seq_dedup and seq_dedup != 'nan' else []
    raw_set = set(items_raw)
    last_item = items_dedup[-1] if items_dedup else None
    prev_item = items_dedup[-2] if len(items_dedup) >= 2 else None
    u_feats = user_feats.get(uid, {})
    hist_len = len(items_raw)
    hist_unique = len(set(items_raw))
    feat = extract_features(uid, target, u_feats, item_feats, item_counts,
                            transitions, last_to_target, item_cat_map,
                            items_raw, items_dedup, raw_set, last_item, hist_len, hist_unique,
                            pair_transitions, last2_to_target, prev_item)
    X_rows.append(feat)
    y_labels.append(1)
    neg_count = 0
    attempts = 0
    while neg_count < neg_per_pos and attempts < neg_per_pos * 3:
        neg_item = all_items[np.random.randint(len(all_items))]
        if neg_item != target and neg_item not in raw_set:
            feat = extract_features(uid, neg_item, u_feats, item_feats, item_counts,
                                    transitions, last_to_target, item_cat_map,
                                    items_raw, items_dedup, raw_set, last_item, hist_len, hist_unique,
                                    pair_transitions, last2_to_target, prev_item)
            X_rows.append(feat)
            y_labels.append(0)
            neg_count += 1
        attempts += 1
    group_sizes.append(1 + neg_count)

X_train = np.array(X_rows)
y_train = np.array(y_labels)
print(f'Samples: {len(X_train)}, Groups: {len(group_sizes)}')

def validate_model(model, sl_val, n=200):
    val_sub = train_df[train_df['item_seq_raw'].apply(seq_len) == sl_val].tail(n)
    scores = []
    for _, row in val_sub.iterrows():
        uid = row['uid']
        seq_raw = str(row['item_seq_raw']).strip()
        seq_dedup = str(row['item_seq_dedup']).strip()
        items_raw = seq_raw.split(',')
        items_dedup = seq_dedup.split(',') if seq_dedup and seq_dedup != 'nan' else []
        raw_set = set(items_raw)
        last_item = items_dedup[-1] if items_dedup else None
        prev_item = items_dedup[-2] if len(items_dedup) >= 2 else None
        u_feats = user_feats.get(uid, {})
        hist_len = len(items_raw)
        hist_unique = len(set(items_raw))
        candidates = set()
        for iid, _ in target_dist.most_common(20):
            candidates.add(iid)
        if last_item and last_item in last_to_target:
            for iid, _ in last_to_target[last_item].most_common(30):
                candidates.add(iid)
        if last_item and last_item in transitions:
            for iid, _ in transitions[last_item].most_common(15):
                candidates.add(iid)
        for item in items_raw:
            candidates.add(item)
        X_cand = []
        cand_list = []
        for iid in candidates:
            feat = extract_features(uid, iid, u_feats, item_feats, item_counts,
                                    transitions, last_to_target, item_cat_map,
                                    items_raw, items_dedup, raw_set, last_item, hist_len, hist_unique,
                                    pair_transitions, last2_to_target, prev_item)
            X_cand.append(feat)
            cand_list.append(iid)
        if not X_cand:
            continue
        X_cand = np.array(X_cand)
        sc = model.predict(X_cand)
        ranked_indices = np.argsort(sc)[::-1]
        preds = [cand_list[i] for i in ranked_indices[:10]]
        scores.append(ndcg_at_k(preds, row['target_iid']))
    return np.mean(scores)

for depth in [20, 24, 28]:
    model = xgb.XGBRanker(
        max_depth=depth, n_estimators=200, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        objective='rank:ndcg', eval_metric='ndcg@10',
        lambdarank_num_pair_per_sample=10,
    )
    model.fit(X_train, y_train, group=group_sizes)
    s1 = validate_model(model, 1)
    s2 = validate_model(model, 2)
    s3 = validate_model(model, 3)
    print(f'depth={depth}: sl1={s1:.4f}, sl2={s2:.4f}, sl3={s3:.4f}')

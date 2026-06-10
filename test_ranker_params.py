"""Test XGBRanker hyperparameters"""
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import xgboost as xgb

base = 'A推荐/A推荐'
train_df = pd.read_csv(f'{base}/train.csv')
user_df = pd.read_csv(f'{base}/user.csv')
item_df = pd.read_csv(f'{base}/item.csv')

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

item_feats = {}
for _, row in item_df.iterrows():
    item_feats[row['iid']] = {'i_cat_01': row['i_cat_01'], 'i_cat_02': row['i_cat_02'],
                               'i_cat_03': row['i_cat_03'], 'i_bucket_01': row['i_bucket_01']}

user_feats = {}
for _, row in user_df.iterrows():
    feats = {}
    for c in user_df.columns:
        if c.startswith('u_cat_'):
            feats[c] = row[c]
    user_feats[row['uid']] = feats

target_dist = Counter(train_df['target_iid'])

def extract_features(uid, iid, u_feats, items_raw, items_dedup, raw_set, last_item, hist_len, hist_unique, prev_item=None):
    i_feats = item_feats.get(iid, {'i_cat_01': 0, 'i_cat_02': 0, 'i_cat_03': 0, 'i_bucket_01': 0})
    feats = [hist_len, hist_unique]
    for c in ['u_cat_01', 'u_cat_02', 'u_cat_03', 'u_cat_04', 'u_cat_05', 'u_cat_06', 'u_cat_07', 'u_cat_08']:
        feats.append(u_feats.get(c, 0))
    feats.extend([i_feats['i_cat_01'], i_feats['i_cat_02'], i_feats['i_cat_03'], i_feats['i_bucket_01']])
    feats.append(np.log1p(item_counts.get(iid, 0)))
    feats.append(1 if iid in raw_set else 0)
    feats.append(1 if iid == last_item else 0)
    feats.append(last_to_target[last_item].get(iid, 0) if last_item and last_item in last_to_target else 0)
    feats.append(transitions[last_item].get(iid, 0) if last_item and last_item in transitions else 0)
    last_cat = item_cat_map.get(last_item, (0, 0, 0)) if last_item else (0, 0, 0)
    cand_cat = (i_feats['i_cat_01'], i_feats['i_cat_02'], i_feats['i_cat_03'])
    feats.append(sum(1 for a, b in zip(cand_cat, last_cat) if a == b) / 3.0)
    hist_cat_match = 0
    if items_dedup:
        for item in items_dedup[-3:]:
            if item_cat_map.get(item, (0, 0, 0)) == cand_cat:
                hist_cat_match = 1
                break
    feats.append(hist_cat_match)
    cooccur_score = 0
    for item in items_dedup[-3:]:
        if item in transitions:
            cooccur_score += transitions[item].get(iid, 0)
    feats.append(cooccur_score)
    pair_trans_score = 0
    pair_l2t_score = 0
    if prev_item and last_item:
        pair = (prev_item, last_item)
        if pair in pair_transitions:
            pair_trans_score = pair_transitions[pair].get(iid, 0)
        if pair in last2_to_target:
            pair_l2t_score = last2_to_target[pair].get(iid, 0)
    feats.extend([pair_trans_score, pair_l2t_score])
    return feats

def seq_len(s):
    if pd.isna(s) or str(s).strip() == '' or str(s).strip() == 'nan':
        return 0
    return len(str(s).split(','))

def ndcg_at_k(ranked_list, target_item, k=10):
    for i, item in enumerate(ranked_list[:k]):
        if item == target_item:
            return 1.0 / np.log2(i + 2)
    return 0.0

# Build training data
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
    feat = extract_features(uid, target, u_feats, items_raw, items_dedup, raw_set, last_item, hist_len, hist_unique, prev_item)
    X_rows.append(feat)
    y_labels.append(1)
    neg_count = 0
    attempts = 0
    while neg_count < neg_per_pos and attempts < neg_per_pos * 3:
        neg_item = all_items[np.random.randint(len(all_items))]
        if neg_item != target and neg_item not in raw_set:
            feat = extract_features(uid, neg_item, u_feats, items_raw, items_dedup, raw_set, last_item, hist_len, hist_unique, prev_item)
            X_rows.append(feat)
            y_labels.append(0)
            neg_count += 1
        attempts += 1
    group_sizes.append(1 + neg_count)

X_train = np.array(X_rows)
y_train = np.array(y_labels)
print(f'Samples: {len(X_train)}, Groups: {len(group_sizes)}')

# Validate function
def validate_model(model, sl_val):
    val_sub = train_df[train_df['item_seq_raw'].apply(seq_len) == sl_val].tail(200)
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
            feat = extract_features(uid, iid, u_feats, items_raw, items_dedup, raw_set, last_item, hist_len, hist_unique, prev_item)
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

# Test configs
configs = [
    {'max_depth': 6, 'n_estimators': 200, 'lr': 0.05},
    {'max_depth': 8, 'n_estimators': 200, 'lr': 0.05},
    {'max_depth': 6, 'n_estimators': 300, 'lr': 0.03},
    {'max_depth': 4, 'n_estimators': 300, 'lr': 0.05},
    {'max_depth': 6, 'n_estimators': 200, 'lr': 0.05, 'min_child_weight': 5},
]

for cfg in configs:
    model = xgb.XGBRanker(
        max_depth=cfg['max_depth'], n_estimators=cfg['n_estimators'], learning_rate=cfg['lr'],
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        objective='rank:ndcg', eval_metric='ndcg@10',
        lambdarank_num_pair_per_sample=10,
        min_child_weight=cfg.get('min_child_weight', 1),
    )
    model.fit(X_train, y_train, group=group_sizes)
    s1 = validate_model(model, 1)
    s2 = validate_model(model, 2)
    print(f'd={cfg["max_depth"]}, n={cfg["n_estimators"]}, lr={cfg["lr"]}, mcw={cfg.get("min_child_weight", 1)}: sl1={s1:.4f}, sl2={s2:.4f}')

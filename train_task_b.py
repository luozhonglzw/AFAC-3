"""
Task B: Sequence Recommendation Baseline
Approach: Popularity + Sequence recency + Item co-occurrence
"""

import os
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_data():
    base = os.path.join(DATA_ROOT, "A推荐", "A推荐")
    train = pd.read_csv(os.path.join(base, "train.csv"))
    test = pd.read_csv(os.path.join(base, "test.csv"))
    user = pd.read_csv(os.path.join(base, "user.csv"))
    item = pd.read_csv(os.path.join(base, "item.csv"))
    sample_sub = pd.read_csv(os.path.join(base, "sample_submission.csv"))
    return train, test, user, item, sample_sub


def parse_seq_counts(s):
    """Parse 'i000968:18,i001455:14' into {iid: count}"""
    if pd.isna(s) or str(s).strip() == "":
        return {}
    result = {}
    for part in str(s).split(","):
        parts = part.split(":")
        if len(parts) == 2:
            result[parts[0]] = int(parts[1])
    return result


def build_popularity(train):
    """Build global item popularity from training data"""
    item_counts = Counter()
    for seq in train["item_seq_raw"].dropna():
        for item_id in str(seq).split(","):
            item_counts[item_id] += 1
    return item_counts


def build_transition_probs(train):
    """Build item transition probabilities (Markov chain)"""
    transitions = defaultdict(Counter)
    for _, row in train.iterrows():
        seq = str(row["item_seq_raw"]).strip()
        if not seq or seq == "nan":
            continue
        items = seq.split(",")
        for i in range(len(items) - 1):
            transitions[items[i]][items[i + 1]] += 1
    return transitions


def build_cooccurrence(train):
    """Build item co-occurrence matrix from sequences"""
    cooccur = defaultdict(Counter)
    for _, row in train.iterrows():
        seq = str(row["item_seq_raw"]).strip()
        if not seq or seq == "nan":
            continue
        items = list(set(seq.split(",")))  # unique items in sequence
        for i in range(len(items)):
            for j in range(len(items)):
                if i != j:
                    cooccur[items[i]][items[j]] += 1
    return cooccur


def build_target_given_last(train):
    """P(target | last item in dedup sequence)"""
    last_to_target = defaultdict(Counter)
    for _, row in train.iterrows():
        target = row["target_iid"]
        seq_dedup = str(row["item_seq_dedup"]).strip()
        if not seq_dedup or seq_dedup == "nan":
            continue
        items = seq_dedup.split(",")
        last_item = items[-1]
        last_to_target[last_item][target] += 1
    return last_to_target


def predict_for_user(
    seq_raw, seq_dedup, item_counts, transitions, cooccur, last_to_target,
    all_items, topk=10
):
    """Generate Top-K predictions for a single user"""
    scores = defaultdict(float)

    # Parse sequences
    items_raw = str(seq_raw).strip().split(",") if pd.notna(seq_raw) and str(seq_raw).strip() else []
    items_dedup = str(seq_dedup).strip().split(",") if pd.notna(seq_dedup) and str(seq_dedup).strip() else []

    # 1. Popularity baseline (low weight)
    for iid, cnt in item_counts.items():
        scores[iid] += 0.001 * cnt

    # 2. Transition probability from last item
    if items_dedup:
        last_item = items_dedup[-1]
        if last_item in transitions:
            for next_item, cnt in transitions[last_item].items():
                scores[next_item] += 0.5 * cnt

        # 3. Last-to-target direct mapping (strong signal)
        if last_item in last_to_target:
            for target, cnt in last_to_target[last_item].items():
                scores[target] += 2.0 * cnt

    # 4. Co-occurrence with recent items (last 3 unique)
    recent = items_dedup[-3:] if len(items_dedup) >= 3 else items_dedup
    for item in recent:
        if item in cooccur:
            for related, cnt in cooccur[item].items():
                scores[related] += 0.3 * cnt

    # 5. Frequency of items in user's own sequence (personal preference)
    user_freq = Counter(items_raw)
    for iid, cnt in user_freq.items():
        scores[iid] += 0.2 * cnt

    # Remove items already in the sequence (optional - they might still be targets)
    # Actually, keep them - the target might be a repeated item

    # Sort by score and return top-K
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top_items = [iid for iid, _ in ranked[:topk]]

    # Pad with popular items if needed
    if len(top_items) < topk:
        for iid, _ in ranked:
            if iid not in top_items:
                top_items.append(iid)
            if len(top_items) >= topk:
                break

    return top_items


def evaluate_hit_rate(train, predictions, k=10):
    """Simple hit rate evaluation on training data (leave-one-out style)"""
    hits = 0
    total = 0
    for _, row in train.head(1000).iterrows():  # sample for speed
        uid = row["uid"]
        target = row["target_iid"]
        if uid in predictions:
            if target in predictions[uid][:k]:
                hits += 1
            total += 1
    return hits / total if total > 0 else 0


def run():
    print("Loading data...")
    train, test, user, item, sample_sub = load_data()

    print("Building item popularity...")
    item_counts = build_popularity(train)

    print("Building transition probabilities...")
    transitions = build_transition_probs(train)

    print("Building co-occurrence matrix...")
    cooccur = build_cooccurrence(train)

    print("Building last-to-target mapping...")
    last_to_target = build_target_given_last(train)

    all_items = set(item["iid"])

    print(f"\nItem popularity top-10: {item_counts.most_common(10)}")
    print(f"Unique items in train: {len(item_counts)}")

    # Evaluate on training data (self-check)
    print("\nEvaluating on training data (sample)...")
    train_preds = {}
    for _, row in train.head(2000).iterrows():
        preds = predict_for_user(
            row["item_seq_raw"], row["item_seq_dedup"],
            item_counts, transitions, cooccur, last_to_target,
            all_items, topk=10
        )
        train_preds[row["uid"]] = preds

    hit_rate = evaluate_hit_rate(train, train_preds, k=1)
    hit_rate_5 = evaluate_hit_rate(train, train_preds, k=5)
    hit_rate_10 = evaluate_hit_rate(train, train_preds, k=10)
    print(f"  Hit@1:  {hit_rate:.4f}")
    print(f"  Hit@5:  {hit_rate_5:.4f}")
    print(f"  Hit@10: {hit_rate_10:.4f}")

    # Generate predictions for test set
    print("\nGenerating test predictions...")
    predictions = {}
    for _, row in test.iterrows():
        preds = predict_for_user(
            row["item_seq_raw"], row["item_seq_dedup"],
            item_counts, transitions, cooccur, last_to_target,
            all_items, topk=10
        )
        predictions[row["uid"]] = preds

    # Save submission
    sub_path = os.path.join(DATA_ROOT, "A2.csv")
    rows = []
    for _, row in sample_sub.iterrows():
        uid = row["uid"]
        preds = predictions.get(uid, list(all_items)[:10])
        rows.append({"uid": uid, "prediction": ",".join(preds)})
    sub = pd.DataFrame(rows)
    sub.to_csv(sub_path, index=False)
    print(f"\nSubmission saved to {sub_path}")
    print(f"Rows: {len(sub)}")
    print(f"Sample: {sub.iloc[0]['prediction'][:80]}...")


if __name__ == "__main__":
    run()

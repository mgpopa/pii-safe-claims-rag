import argparse, json, faiss, numpy as np, pandas as pd, time, os

def build_eval_set(df, n=200, seed=42):
    rng = np.random.RandomState(seed)
    sample = df.sample(min(n, len(df)), random_state=seed)
    qs = []
    for _, row in sample.iterrows():
        name = str(row["contact_name"]).split()[0]
        policy = row["policy_id"]
        city = row["city"]
        desc = row["loss_description"]
        q_pii = f"Email {name} about policy {policy} claim in {city}"
        q_gen = f"{desc}"
        qs.append((q_pii, row["claim_id"], "pii"))
        qs.append((q_gen, row["claim_id"], "generic"))
    return qs

def recall_curve(index, model, queries, meta, ks):
    import time
    rows = []
    for k in ks:
        hits = 0
        latencies = []
        for q, gold_claim, bucket in queries:
            t0 = time.time()
            emb = model.encode([q], normalize_embeddings=True)
            scores, idxs = index.search(emb.astype(np.float32), int(k))
            latencies.append((time.time()-t0)*1000.0)
            idxs = idxs[0].tolist()
            claim_ids = [meta[str(i)]["claim_id"] for i in idxs]
            hits += int(gold_claim in claim_ids)
        recall = hits / len(queries) if queries else 0.0
        rows.append({"k": int(k), "recall": recall, "avg_latency_ms": float(np.mean(latencies)), "p95_latency_ms": float(np.percentile(latencies,95))})
    return pd.DataFrame(rows)

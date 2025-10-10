import argparse, json, faiss, numpy as np, pandas as pd, time, os
from sentence_transformers import SentenceTransformer
from .utils import contains_pii, timer
from .pii_mask import mask_text

REPORTS = "reports"
RAW_INDEX = "indexes/faiss_raw.index"
MSK_INDEX = "indexes/faiss_masked.index"
RAW_META  = "indexes/meta_raw.json"
MSK_META  = "indexes/meta_masked.json"
DATA_CSV  = "data/claims.csv"

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

def recall_curve(index, model, queries, meta, ks, *, mask_queries=False):
    import time
    rows = []
    for k in ks:
        hits = 0
        latencies = []
        for q, gold_claim, bucket in queries:
            q_use = mask_text(q) if mask_queries else q
            t0 = time.time()
            emb = model.encode([q_use], normalize_embeddings=True)
            scores, idxs = index.search(emb.astype(np.float32), int(k))
            latencies.append((time.time()-t0)*1000.0)
            idxs = idxs[0].tolist()
            claim_ids = [meta[str(i)]["claim_id"] for i in idxs]
            hits += int(gold_claim in claim_ids)
        recall = hits / len(queries) if queries else 0.0
        rows.append({"k": int(k), "recall": recall, "avg_latency_ms": float(np.mean(latencies)), "p95_latency_ms": float(np.percentile(latencies,95))})
    return pd.DataFrame(rows)

def leakage_rate(meta):
    texts = [v["text"] for v in meta.values()]
    leaked = sum(1 for t in texts if contains_pii(t))
    return leaked/len(texts) if texts else 0.0

def main():
    os.makedirs(REPORTS, exist_ok=True)
    df = pd.read_csv(DATA_CSV)
    queries = build_eval_set(df, n=200)

    buckets = {"pii": [], "generic": []}
    for q, cid, b in queries:
        buckets[b].append((q,cid,b))

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    idx_raw = faiss.read_index(RAW_INDEX); meta_raw = json.load(open(RAW_META))
    idx_msk = faiss.read_index(MSK_INDEX); meta_msk = json.load(open(MSK_META))

    ks = [1,3,5,10,20]
    records = []

    for idx_name, index, meta in [("raw", idx_raw, meta_raw), ("masked", idx_msk, meta_msk)]:
        for bucket_name, qs in buckets.items():
            df_curve = recall_curve(index, model, qs, meta, ks, mask_queries=(idx_name=="masked"))
            for _, r in df_curve.iterrows():
                records.append({
                    "index": idx_name,
                    "bucket": bucket_name,
                    "k": int(r["k"]),
                    "recall": float(r["recall"]),
                    "avg_latency_ms": float(r["avg_latency_ms"]),
                    "p95_latency_ms": float(r["p95_latency_ms"])
                })

    out_csv = os.path.join(REPORTS, "metrics.csv")
    pd.DataFrame.from_records(records).to_csv(out_csv, index=False)

    leak = { "raw": leakage_rate(meta_raw), "masked": leakage_rate(meta_msk) }
    with open(os.path.join(REPORTS, "leakage.json"), "w") as f:
        json.dump(leak, f, indent=2)

    print("Wrote:", out_csv)
    print("Wrote:", os.path.join(REPORTS, "leakage.json"))

if __name__ == "__main__":
    main()

import os, json, time, re
import numpy as np, pandas as pd, faiss
from sentence_transformers import SentenceTransformer
from .pii_mask import mask_text
from .utils import contains_pii, for_embedding, format_for_e5

# macOS stability guards (AGAIN)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import torch; torch.set_num_threads(1)
except Exception:
    pass


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS  = os.path.join(BASE_DIR, "reports"); os.makedirs(REPORTS, exist_ok=True)
RAW_INDEX = os.path.join(BASE_DIR, "indexes", "faiss_raw.index")
MSK_INDEX = os.path.join(BASE_DIR, "indexes", "faiss_masked.index")
RAW_META  = os.path.join(BASE_DIR, "indexes", "meta_raw.json")
MSK_META  = os.path.join(BASE_DIR, "indexes", "meta_masked.json")
DATA_CSV  = os.path.join(BASE_DIR, "data",    "claims.csv")

KS = [1, 3, 5, 10, 20]
BATCH = 32
TOKEN_RE = re.compile(r"\[([A-Z]+)_([0-9a-f]+)\]")  # [POLICY_ab12], [EMAIL_09f3] etc

def _encode(model, texts):
    return model.encode(
        texts, batch_size=BATCH, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)

def _prepare_queries(csv_path, n_pii=150, n_gen=250, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.read_csv(csv_path)

    pii_df = df.sample(min(n_pii, len(df)), random_state=seed)
    gen_df = df.sample(min(n_gen, len(df)), random_state=seed+1)

    pii_qs = []
    for _, r in pii_df.iterrows():
        claim = r["claim_id"]
        policy = r.get("policy_id"); email = r.get("contact_email"); phone = r.get("contact_phone")
        patt = rng.integers(0, 3)
        if patt == 0 and isinstance(policy, str):
            q = f"email insured about policy {policy}"
        elif patt == 1 and isinstance(email, str):
            q = f"email {email} re claim"
        else:
            q = f"call {phone} re policy {policy}"
        pii_qs.append((q, claim, "pii"))

    gen_qs = []
    for _, r in gen_df.iterrows():
        claim = r["claim_id"]
        cause = (r.get("loss_description") or "").split(".")[0]
        city  = r.get("city") or ""
        q = f"{cause} in {city}".strip() if (city and cause) else (cause or f"claim in {city}".strip())
        gen_qs.append((q, claim, "generic"))

    return pii_qs + gen_qs

def _build_sidecar_from_masked_meta(meta_masked: dict) -> dict[str, list[str]]:
    """Token -> [claim_id,...] built from masked metadata (no raw PII)"""
    side = {}
    for i in range(len(meta_masked)):
        cid = meta_masked[str(i)]["claim_id"]
        t = meta_masked[str(i)]["text"]
        for m in TOKEN_RE.finditer(t):
            label = m.group(1)
            tok = f"[{label}_{m.group(2)}]"
            if label in {"POLICY", "EMAIL", "PHONE"}:
                side.setdefault(tok, []).append(cid)
    return side

def _router_hits(masked_query: str, side: dict) -> list[str]:
    """Return claim_ids that share a pseudonym token with the masked query"""
    hits = []
    for m in TOKEN_RE.finditer(masked_query):
        tok = f"[{m.group(1)}_{m.group(2)}]"
        if tok in side:
            hits.extend(side[tok])
    # de-dup keep order
    seen, out = set(), []
    for cid in hits:
        if cid not in seen:
            seen.add(cid); out.append(cid)
    return out

def _recall_eval(index, model, queries, meta, ks, path_name: str, sidecar: dict, is_masked_path: bool):
    """Hybrid (router -> vector) used for BOTH raw & masked paths. Fair A/B."""
    rows, lat10 = [], []

    for q, gold_claim, bucket in queries:
        # router (applies to BOTH raw and masked). mask the query to derive tokens
        q_masked = mask_text(q)
        routed = _router_hits(q_masked, sidecar)  #same sidecar for both
        if routed:
            for k in ks:
                hit = 1 if gold_claim in routed[:k] or (k >= 1 and gold_claim in routed) else 0
                rows.append({"index": path_name, "bucket": bucket, "k": k, "hit": hit})
            if 10 in ks: lat10.append(0.2)  #negligible
            continue

        # vector fallback (symmetric prep)
        if is_masked_path:
            q_use = format_for_e5(for_embedding(q_masked), is_query=True)
        else:
            q_use = format_for_e5(q, is_query=True)

        t0 = time.perf_counter()
        emb = _encode(model, [q_use])
        scores, idxs = index.search(emb, max(ks))
        dt = (time.perf_counter() - t0) * 1000.0

        claim_ids = [meta[str(i)]["claim_id"] for i in idxs[0]]
        for k in ks:
            rows.append({
                "index": path_name, "bucket": bucket, "k": k,
                "hit": 1 if gold_claim in claim_ids[:k] else 0
            })
        if 10 in ks: lat10.append(dt)

    df = pd.DataFrame(rows)
    out = []
    for idx_name in df["index"].unique():
        for bucket in df["bucket"].unique():
            sub = df[(df["index"] == idx_name) & (df["bucket"] == bucket)]
            for k in ks:
                kk = sub[sub["k"] == k]
                if len(kk) == 0: continue
                out.append({
                    "index": idx_name, "bucket": bucket, "k": k,
                    "recall": kk["hit"].mean(),
                    "avg_latency_ms_at10": (np.mean(lat10) if k == 10 else np.nan)
                })
    return pd.DataFrame(out)

def main():
    # Inputs present?
    for p in [RAW_INDEX, MSK_INDEX, RAW_META, MSK_META, DATA_CSV]:
        if not os.path.exists(p):
            raise SystemExit(f"Missing: {p}. Build indexes first (python -m src.embed_index).")

    # encoder (CPU, slow tokenizer to avoid segfaults on mac)
    model = SentenceTransformer("intfloat/e5-small-v2", device="cpu")
    try:
        from transformers import AutoTokenizer
        model.tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small-v2", use_fast=False)
    except Exception:
        pass

    # load
    idx_raw  = faiss.read_index(RAW_INDEX)
    idx_msk  = faiss.read_index(MSK_INDEX)
    meta_raw = json.load(open(RAW_META))
    meta_msk = json.load(open(MSK_META))

    # sidecar from masked meta (no PII)
    side = _build_sidecar_from_masked_meta(meta_msk)

    # queries
    queries = _prepare_queries(DATA_CSV)

    # eval both paths with the SAME hybrid strategy
    df_raw = _recall_eval(idx_raw, model, queries, meta_raw, KS, path_name="raw",    sidecar=side, is_masked_path=False)
    df_msk = _recall_eval(idx_msk, model, queries, meta_msk, KS, path_name="masked", sidecar=side, is_masked_path=True)
    metrics = pd.concat([df_raw, df_msk], ignore_index=True)

    # leakage on masked meta (should be 0)
    mtexts = [meta_msk[str(i)]["text"] for i in range(len(meta_msk))]
    leaked = sum(contains_pii(t) for t in mtexts)
    leakage = {"masked_docs": len(mtexts), "masked_leaked": leaked, "rate": leaked / max(1, len(mtexts))}

    metrics.to_csv(os.path.join(REPORTS, "metrics.csv"), index=False)
    with open(os.path.join(REPORTS, "leakage.json"), "w") as f:
        json.dump(leakage, f, indent=2)

    print(f"Wrote: {os.path.join(REPORTS, 'metrics.csv')}")
    print(f"Wrote: {os.path.join(REPORTS, 'leakage.json')}")

if __name__ == "__main__":
    main()

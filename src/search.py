import argparse, json, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from .pii_mask import mask_text
from .utils import contains_pii, for_embedding

RAW_INDEX = "indexes/faiss_raw.index"
MSK_INDEX = "indexes/faiss_masked.index"
RAW_META  = "indexes/meta_raw.json"
MSK_META  = "indexes/meta_masked.json"

def load_index(path):
    return faiss.read_index(path)

def main(q, k, use_raw=False):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if use_raw:
        q_text = q # raw index gets a raw query
        index = faiss.read_index(RAW_INDEX); meta = json.load(open(RAW_META))
    else:
        q_text = for_embedding(mask_text(q)) # masked index gets a masked query
        index = faiss.read_index(MSK_INDEX); meta = json.load(open(MSK_META))

    emb = model.encode([q], normalize_embeddings=True).astype(np.float32)
    scores, idxs = index.search(emb, k)
    idxs = idxs[0].tolist(); scores = scores[0].tolist()

    # Always fetch raw text for display (by id) so PII never lives in masked vectors
    raw_meta = json.load(open(RAW_META))

    print("\nQuery:", q)
    print("Index:", "RAW" if use_raw else "MASKED (PII-safe)")
    for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
        claim_id = meta[str(i)]["claim_id"]
        masked   = meta[str(i)]["text"]
        raw_text = raw_meta[str(i)]["text"]
        print("\n#", rank, f"score={s:.3f}")
        print("Claim:", claim_id)
        print("Masked snippet:", (masked[:240].replace("\n"," ") + ("..." if len(masked)>240 else "")))
        print("Raw for display:", (raw_text[:240].replace("\n"," ") + ("..." if len(raw_text)>240 else "")))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="user query text")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--use_raw", action="store_true")
    args = ap.parse_args()
    main(args.q, args.k, args.use_raw)

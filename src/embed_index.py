import os, json, numpy as np, faiss, pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pii_mask import mask_text
from utils import timer, contains_pii

DATA_CSV = "data/claims.csv"
IDX_DIR = "indexes"
RAW_INDEX = os.path.join(IDX_DIR, "faiss_raw.index")
MSK_INDEX = os.path.join(IDX_DIR, "faiss_masked.index")
RAW_META = os.path.join(IDX_DIR, "meta_raw.json")
MSK_META = os.path.join(IDX_DIR, "meta_masked.json")

def build():
    os.makedirs(IDX_DIR, exist_ok=True)
    if not os.path.exists(DATA_CSV):
        raise SystemExit(f"Missing {DATA_CSV}. Run: python src/data_gen.py --n 5000")

    df = pd.read_csv(DATA_CSV)
    raw_texts, masked_texts = [], []
    meta_raw, meta_masked = {}, {}

    print("Masking notes...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        note = str(row["adjuster_note"]).strip()
        masked = mask_text(note)
        claim_id = row["claim_id"]
        raw_texts.append(note)
        masked_texts.append(masked)
        meta_raw[i] = {"claim_id": claim_id, "text": note}
        meta_masked[i] = {"claim_id": claim_id, "text": masked}

    print("Encoding (all-MiniLM-L6-v2)...")
    enc_t = timer()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb_raw = model.encode(raw_texts, batch_size=256, show_progress_bar=True,
                           convert_to_numpy=True, normalize_embeddings=True)
    emb_masked = model.encode(masked_texts, batch_size=256, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=True)
    print(f"Encoded in {enc_t():.2f}s -> shape {emb_raw.shape}")

    # build cosine-sim indices (IP on normalized vectors == cosine)
    d = emb_raw.shape[1]
    index_raw = faiss.IndexFlatIP(d)
    index_msk = faiss.IndexFlatIP(d)
    index_raw.add(emb_raw.astype(np.float32))
    index_msk.add(emb_masked.astype(np.float32))

    faiss.write_index(index_raw, RAW_INDEX)
    faiss.write_index(index_msk, MSK_INDEX)
    with open(RAW_META, "w") as f: json.dump(meta_raw, f)
    with open(MSK_META, "w") as f: json.dump(meta_masked, f)

    leaked = sum(1 for t in masked_texts if contains_pii(t))
    print(f"Leakage check (masked docs containing PII): {leaked} / {len(masked_texts)} (aim 0)")
    print("Done.")

if __name__ == "__main__":
    build()

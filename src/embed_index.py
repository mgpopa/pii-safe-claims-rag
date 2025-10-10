import os, json, numpy as np, faiss, pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from .utils import for_embedding

# Use package-relative imports
from .pii_mask import mask_text
from .utils import contains_pii as contains_pii_utils

DATA_CSV = "data/claims.csv"
IDX_DIR = "indexes"
RAW_INDEX = os.path.join(IDX_DIR, "faiss_raw.index")
MSK_INDEX = os.path.join(IDX_DIR, "faiss_masked.index")
RAW_META = os.path.join(IDX_DIR, "meta_raw.json")
MSK_META = os.path.join(IDX_DIR, "meta_masked.json")

def build():
    os.makedirs(IDX_DIR, exist_ok=True)
    if not os.path.exists(DATA_CSV):
        raise SystemExit(f"Missing {DATA_CSV}. Run: python -m src.data_gen --n 5000")

    df = pd.read_csv(DATA_CSV)

    raw_texts, masked_texts = [], []
    meta_raw, meta_masked = {}, {}

    print("Masking notes...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        note = str(row["adjuster_note"]).strip()
        masked = mask_text(note)
        raw_texts.append(note)
        masked_texts.append(masked)
        meta_raw[i] = {"claim_id": row["claim_id"], "text": note}
        meta_masked[i] = {"claim_id": row["claim_id"], "text": masked}

    ph_count = sum(("[EMAIL]" in t) or ("[PHONE]" in t) or ("[POLICY_ID]" in t) for t in masked_texts)
    print(f"[DEBUG] placeholders in masked_texts: {ph_count}/{len(masked_texts)}")

    # prep texts for embedding
    # RAW: embed as-is
    # MASKED: convert placeholders to plain words so the model "gets" the meaning
    masked_texts_embed = [for_embedding(t) for t in masked_texts]

    print("Encoding (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    emb_raw = model.encode(
        raw_texts, batch_size=128, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True
    ).astype(np.float32)

    emb_masked = model.encode(
        masked_texts_embed, batch_size=128, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True
    ).astype(np.float32)

    # build FAISS indexes
    d = emb_raw.shape[1]
    index_raw = faiss.IndexFlatIP(d)
    index_msk = faiss.IndexFlatIP(d)
    index_raw.add(emb_raw)
    index_msk.add(emb_masked)

    # Persist
    faiss.write_index(index_raw, RAW_INDEX)
    faiss.write_index(index_msk, MSK_INDEX)
    with open(RAW_META, "w") as f: json.dump(meta_raw, f)
    with open(MSK_META, "w") as f: json.dump(meta_masked, f)

    # leakage sanity
    leaked = sum(contains_pii_utils(t) for t in masked_texts)
    print(f"Leakage check (masked docs containing PII): {leaked} / {len(masked_texts)} (aim 0)")
    print("Done.")

if __name__ == "__main__":
    build()
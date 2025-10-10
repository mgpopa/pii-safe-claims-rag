import os, json, re
import numpy as np, pandas as pd, faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from .pii_mask import mask_text
from .utils import contains_pii, for_embedding, format_for_e5

# avoid tokenizer/threads segfaults on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass


DATA_CSV = "data/claims.csv"
IDX_DIR  = "indexes"
RAW_INDEX = os.path.join(IDX_DIR, "faiss_raw.index")
MSK_INDEX = os.path.join(IDX_DIR, "faiss_masked.index")
RAW_META  = os.path.join(IDX_DIR, "meta_raw.json")
MSK_META  = os.path.join(IDX_DIR, "meta_masked.json")
SIDECAR   = os.path.join(IDX_DIR, "sidecar.json")  # pseudonym -> [claim_id, ...]

# pseudonym token pattern like [POLICY_ab12], [EMAIL_09f3]
TOKEN_RE = re.compile(r"\[([A-Z]+)_([0-9a-f]+)\]")

# small, non-PII tags that help retrieval
def _mk_struct_tail(row: pd.Series) -> str:
    city = (str(row.get("city") or "")).lower().strip().replace(" ", "_")
    lob  = (str(row.get("lob")  or "")).lower().strip().replace(" ", "_")
    return f" tags: city:{city} lob:{lob}"

def build():
    os.makedirs(IDX_DIR, exist_ok=True)
    if not os.path.exists(DATA_CSV):
        raise SystemExit("Missing data/claims.csv. Run: python -m src.data_gen --n 5000")

    df = pd.read_csv(DATA_CSV).reset_index(drop=True)

    # mask notes; collect raw/masked + metadata
    raw_texts, masked_texts = [], []
    meta_raw, meta_masked = {}, {}
    print("Masking notes...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        note   = str(row["adjuster_note"]).strip()
        masked = mask_text(note)
        raw_texts.append(note)
        masked_texts.append(masked)
        meta_raw[i]    = {"claim_id": row["claim_id"], "text": note}
        meta_masked[i] = {"claim_id": row["claim_id"], "text": masked}

    #Sidecar: map pseudonym tokens -> list of claim_ids (no raw PII stored)
    side: dict[str, list[str]] = {}
    for i, t in enumerate(masked_texts):
        cid = meta_raw[i]["claim_id"]
        for m in TOKEN_RE.finditer(t):
            label = m.group(1)      # policy / email / phones / ...
            tok   = m.group(0)      # full token, e.g. [POLICY_ab12]
            if label in {"POLICY", "EMAIL", "PHONE"}:
                side.setdefault(tok, []).append(cid)
    with open(SIDECAR, "w") as f:
        json.dump(side, f)
    print(f"Sidecar tokens stored: {len(side)}")

    # prep passages (use e5 with instruction prefixes)
    raw_passages = [format_for_e5(t, is_query=False) for t in raw_texts]
    masked_passages = [
        format_for_e5(for_embedding(masked_texts[i]) + _mk_struct_tail(df.iloc[i]), is_query=False)
        for i in range(len(masked_texts))
    ]

    # encode (CPU, smaller batch, slow tokenizer to dodge segfaults)
    print("Encoding with intfloat/e5-small-v2 (CPU)...")
    model = SentenceTransformer("intfloat/e5-small-v2", device="cpu")

    # force slow tokenizer (pure Python)
    try:
        from transformers import AutoTokenizer
        model.tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small-v2", use_fast=False)
    except Exception:
        pass

    BATCH = 32  # im keeping this modest for stability
    emb_raw = model.encode(
        raw_passages, batch_size=BATCH, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True
    ).astype(np.float32)
    emb_masked = model.encode(
        masked_passages, batch_size=BATCH, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True
    ).astype(np.float32)

    # build FAISS (cosine via inner product on normalized vectors)
    d = emb_raw.shape[1]
    index_raw = faiss.IndexFlatIP(d)
    index_msk = faiss.IndexFlatIP(d)
    index_raw.add(emb_raw)
    index_msk.add(emb_masked)

    # poersist
    faiss.write_index(index_raw, RAW_INDEX)
    faiss.write_index(index_msk, MSK_INDEX)
    with open(RAW_META, "w") as f: json.dump(meta_raw, f)
    with open(MSK_META, "w") as f: json.dump(meta_masked, f)

    # leakage sanity
    leaked = sum(contains_pii(t) for t in masked_texts)
    print(f"Leakage check (masked docs containing PII): {leaked} / {len(masked_texts)}")
    print("Done.")

if __name__ == "__main__":
    build()

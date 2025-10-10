# Meet the Cast - What Each File Does (and One Juicy Detail)

You don't need a 30-page architecture doc. Here's the **PII-Safe Claims Notes Retrieval - PLUS Edition** repo in plain English - tight, honest, and tuned for folks who Google *"LLM semantic search insurance"* at 11pm.

---

## `data_gen.py` - fake it till you make it
Spins up a synthetic claims dataset: policy IDs, emails, phones, cities, loss descriptions, and short adjuster notes.  
**Why:** Safely test RAG, vectorization, and PII masking without touching real customer data.  
**Cool fact:** Every run is **seeded**, so you can regenerate the same chaos for A/B tests and consistent plots.

---

## `pii_mask.py` - the privacy paint roller
Zaps obvious PII with regex (emails, phones, policy IDs) and scrubs names/orgs via spaCy NER; keeps cities for context. Replaces sensitive bits with stable pseudonyms like `[PERSON_7a1c]`.  
**Why:** The **vector index never sees PII**.  
**Cool fact:** Pseudonyms use salted hashes, so "John Smith" consistently maps to the same token-retrieval stays stable.

---

## `utils.py` - glue, but the useful kind
Small helpers with big payoff:
- `for_embedding()` turns tokens like `[PERSON_ab12]` into model-friendly text ("person token ab12") so SentenceTransformers actually learns something.  
- `format_for_e5()` adds the `"query:"`/`"passage:"` prefixes **e5** models adore.  
- `contains_pii()` checks leakage **without** getting fooled by placeholders.  
**Cool fact:** Phone detection uses `phonenumbers` plus sanity filters so currency amounts and random digits don't trigger false positives.

---

## `embed_index.py` - turning stories into vectors (twice)
Builds **two FAISS indexes**: one from raw notes, one from masked notes. Uses `intfloat/e5-small-v2` on CPU and writes tidy metadata JSONs.  
**Why:** Clean A/B testing of **raw vs masked** semantic search.  
**Cool fact:** Writes a **sidecar** that maps pseudonym tokens (e.g., `[POLICY_xxxx]`) to claim IDs. No raw PII stored, but exact-key lookups still work.

---

## `eval_plus.py` - the scoreboard you can take to a meeting
Generates PII-leaning and generic queries, runs both **raw** and **masked** paths, and logs **Recall@k** + **latency** to `reports/metrics.csv` and leakage to `reports/leakage.json`.  
**Why:** You need numbers, not vibes.  
**Cool fact:** Evaluation is **fair**-both paths use the same strategy (policy-token router when uniquely identifying, then vector fallback). No cheating.

---

## `plots.py` - pictures or it didn't happen
Turns metrics into the five money slides: **Before vs After (Recall@10)**, **Recall curves (PII + generic)**, **Latency**, and **Leakage**.  
**Why:** Execs don't read CSVs.  
**Cool fact:** Paths are repo-anchored and y-limits are forced, so "zero leakage" doesn't render as invisible.

---

## `search.py` - the CLI that proves the point
Tiny command-line tool to query either index. Shows the **masked snippet** (what the index saw) and the **raw note** (what a human reads).  
**Why:** Instant trust demo for privacy-by-design.  
**Cool fact:** Prints a "PII present?" flag for the masked snippet-great for catching rules you forgot.

---

> **TL;DR:** This repo shows how to do **PII-safe semantic search for insurance claims** with **LLMs**, **FAISS**, **spaCy NER**, and **SentenceTransformers**. Hard numbers, cool plots, and a demo that doesn't terrify compliance.

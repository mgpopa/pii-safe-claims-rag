import numpy as np

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

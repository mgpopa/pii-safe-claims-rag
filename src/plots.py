import os, json
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS  = os.path.join(BASE_DIR, "reports")
METRICS  = os.path.join(REPORTS, "metrics.csv")
LEAKAGE  = os.path.join(REPORTS, "leakage.json")
RAW_META = os.path.join(BASE_DIR, "indexes", "meta_raw.json")
MSK_META = os.path.join(BASE_DIR, "indexes", "meta_masked.json")

os.makedirs(REPORTS, exist_ok=True)

def _read_metrics():
    if not os.path.exists(METRICS):
        raise SystemExit(f"Missing {METRICS}. Run: python -m src.eval_plus")
    df = pd.read_csv(METRICS)
    # unify column name
    if "avg_latency_ms" not in df.columns and "avg_latency_ms_at10" in df.columns:
        df["avg_latency_ms"] = df["avg_latency_ms_at10"]
    return df

def _read_leakage():
    if os.path.exists(LEAKAGE):
        return json.load(open(LEAKAGE))
    return {"masked_docs": 0, "masked_leaked": 0, "rate": 0.0}

def _savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def _force_ylim(ax, values):
    ymax = max([v for v in values if pd.notna(v)] + [0.001])
    ax.set_ylim(0, max(0.05, ymax * 1.15))  # make zero bars visible

def main():
    df = _read_metrics()
    leakage = _read_leakage()

    # Before vs After Recall@10
    df10 = df[df["k"] == 10].copy()
    pivot = df10.pivot_table(index="bucket", columns="index", values="recall", aggfunc="mean")
    ax = pivot.plot(kind="bar")
    ax.set_ylabel("Recall@10")
    ax.set_title("Before vs After Recall@10 - raw vs masked")
    _force_ylim(ax, pivot.values.flatten())
    _savefig(os.path.join(REPORTS, "before_after_recall_at10.png"))

    # Recall Curve for PII-leaning
    pii = df[df["bucket"] == "pii"].pivot_table(index="k", columns="index", values="recall", aggfunc="mean")
    ax = pii.plot(marker="o")
    ax.set_ylabel("Recall@k")
    ax.set_title("Recall vs k - PII-leaning queries")
    _force_ylim(ax, pii.values.flatten())
    _savefig(os.path.join(REPORTS, "recall_curve_pii.png"))

    # Recall Curve for Generic queries
    gen = df[df["bucket"] == "generic"].pivot_table(index="k", columns="index", values="recall", aggfunc="mean")
    ax = gen.plot(marker="o")
    ax.set_ylabel("Recall@k")
    ax.set_title("Recall vs k - Generic queries")
    _force_ylim(ax, gen.values.flatten())
    _savefig(os.path.join(REPORTS, "recall_curve_generic.png"))

    # Latency @k=10
    lat = df10.pivot_table(index="bucket", columns="index", values="avg_latency_ms", aggfunc="mean")
    ax = lat.plot(kind="bar")
    ax.set_ylabel("Avg latency (ms) @k=10")
    ax.set_title("Average query latency @k=10")
    _force_ylim(ax, lat.values.flatten())
    _savefig(os.path.join(REPORTS, "latency_at10.png"))

    # Leakage bar (masked, and raw if computed)
    bars = {"masked": leakage.get("rate", 0.0)}
    try:
        from src.utils import contains_pii
        if os.path.exists(RAW_META):
            raw = json.load(open(RAW_META))
            rtexts = [raw[str(i)]["text"] for i in range(len(raw))]
            bars["raw"] = sum(contains_pii(t) for t in rtexts) / max(1, len(rtexts))
    except Exception:
        pass
    labels = list(bars.keys()); values = [bars[k] for k in labels]
    ax = pd.Series(values, index=labels).plot(kind="bar")
    ax.set_ylabel("Leakage rate")
    ax.set_title("PII leakage (lower is better)")
    _force_ylim(ax, values)
    _savefig(os.path.join(REPORTS, "leakage.png"))

    print(f"Charts written to {REPORTS}")

if __name__ == "__main__":
    main()

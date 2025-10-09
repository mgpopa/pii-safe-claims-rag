import os, json, pandas as pd
import matplotlib.pyplot as plt

REPORTS = "reports"

def plot_recall_at_k(df, bucket, out_png):
    sub = df[(df["bucket"]==bucket)]
    ks = sorted(sub["k"].unique())
    raw = [sub[(sub["index"]=="raw") & (sub["k"]==k)]["recall"].mean() for k in ks]
    msk = [sub[(sub["index"]=="masked") & (sub["k"]==k)]["recall"].mean() for k in ks]

    plt.figure()
    plt.plot(ks, raw, marker="o", label="raw")
    plt.plot(ks, msk, marker="o", label="masked")
    plt.xlabel("k")
    plt.ylabel("Recall@k")
    plt.title(f"Recall vs k ({bucket} queries)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def bar_recall_k(df, k, out_png):
    sub = df[df["k"]==k]
    buckets = ["pii", "generic"]
    x = range(len(buckets))
    raw = [sub[(sub["index"]=="raw") & (sub["bucket"]==b)]["recall"].mean() for b in buckets]
    msk = [sub[(sub["index"]=="masked") & (sub["bucket"]==b)]["recall"].mean() for b in buckets]

    plt.figure()
    width = 0.35
    x_raw = [i - width/2 for i in x]
    x_msk = [i + width/2 for i in x]
    plt.bar(x_raw, raw, width, label="raw")
    plt.bar(x_msk, msk, width, label="masked")
    plt.xticks(list(x), buckets)
    plt.ylabel(f"Recall@{k}")
    plt.title(f"Before vs After (Recall@{k})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def latency_bars(df, k, out_png):
    sub = df[df["k"]==k]
    buckets = ["pii", "generic"]
    raw = [sub[(sub["index"]=="raw") & (sub["bucket"]==b)]["avg_latency_ms"].mean() for b in buckets]
    msk = [sub[(sub["index"]=="masked") & (sub["bucket"]==b)]["avg_latency_ms"].mean() for b in buckets]

    plt.figure()
    width = 0.35
    x = range(len(buckets))
    x_raw = [i - width/2 for i in x]
    x_msk = [i + width/2 for i in x]
    plt.bar(x_raw, raw, width, label="raw")
    plt.bar(x_msk, msk, width, label="masked")
    plt.xticks(list(x), buckets)
    plt.ylabel("Avg query latency (ms)")
    plt.title(f"Latency (k={k})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def leakage_bar(leak_json, out_png):
    with open(leak_json) as f:
        leak = json.load(f)
    plt.figure()
    labels = ["raw", "masked"]
    vals = [leak.get("raw",0), leak.get("masked",0)]
    plt.bar(labels, vals)
    plt.ylabel("Leakage rate")
    plt.title("PII Leakage (lower is better)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def main():
    metrics_csv = os.path.join(REPORTS, "metrics.csv")
    df = pd.read_csv(metrics_csv)

    plot_recall_at_k(df, "pii", os.path.join(REPORTS, "recall_curve_pii.png"))
    plot_recall_at_k(df, "generic", os.path.join(REPORTS, "recall_curve_generic.png"))
    bar_recall_k(df, 10, os.path.join(REPORTS, "before_after_recall_at10.png"))
    latency_bars(df, 10, os.path.join(REPORTS, "latency_at10.png"))
    leakage_bar(os.path.join(REPORTS, "leakage.json"), os.path.join(REPORTS, "leakage.png"))
    print("Wrote charts to reports/.")

if __name__ == "__main__":
    main()

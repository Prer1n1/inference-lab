"""Plot benchmark results.

    python bench/plot.py --out curve.png clean=labs/05_vllm_baseline/results/clean.jsonl
    python bench/plot.py --out cmp.png --hollow-above 100 clean=a.jsonl cached=b.jsonl

Left panel : output tokens/s vs load (users in flight, or requests/s for open-loop runs)
Right panel: the tradeoff curve, tokens/s vs p99 inter-token latency (up and to the left is better)
Points above --hollow-above are drawn hollow: use it for levels you do not trust (e.g. a known client cap).
Rows from several runs at the same level are averaged for throughput.
"""
import argparse
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    by_level = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            level = r.get("level", r.get("concurrency"))
            by_level[level].append(r)
    pts = []
    for level in sorted(by_level):
        rows = by_level[level]
        mean = lambda k: sum(r[k] for r in rows if r.get(k) is not None) / max(1, sum(r.get(k) is not None for r in rows))  # noqa: E731
        pts.append({"level": level, "tps": mean("out_tok_per_s"), "itl99": mean("itl_ms_p99"),
                    "itl50": mean("itl_ms_p50")})
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("series", nargs="+", help="label=path.jsonl")
    ap.add_argument("--out", default="curve.png")
    ap.add_argument("--title", default="")
    ap.add_argument("--hollow-above", type=float, default=None)
    a = ap.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for item in a.series:
        label, path = item.split("=", 1)
        pts = load(path)
        line, = ax1.plot([p["level"] for p in pts], [p["tps"] for p in pts], "-", label=label)
        ax2.plot([p["itl99"] for p in pts], [p["tps"] for p in pts], "-", color=line.get_color(), label=label)
        for p in pts:
            hollow = a.hollow_above is not None and p["level"] > a.hollow_above
            kw = dict(marker="o", ls="", color=line.get_color(), mfc="white" if hollow else line.get_color())
            ax1.plot(p["level"], p["tps"], **kw)
            ax2.plot(p["itl99"], p["tps"], **kw)
            ax2.annotate(f"{p['level']:g}", (p["itl99"], p["tps"]), fontsize=7, xytext=(3, -9),
                         textcoords="offset points")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("load (users in flight)")
    ax1.set_ylabel("output tokens / s")
    ax1.grid(alpha=0.3)
    ax2.set_xscale("log")
    ax2.set_xlabel("p99 inter-token latency (ms)")
    ax2.set_ylabel("output tokens / s")
    ax2.grid(alpha=0.3)
    ax1.legend()
    note = f"  (hollow = above {a.hollow_above:g}, not trusted)" if a.hollow_above is not None else ""
    fig.suptitle((a.title or "Throughput vs latency") + note)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

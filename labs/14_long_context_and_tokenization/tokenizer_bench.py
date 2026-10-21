"""Lab 14, Part A: how expensive is tokenization, does it matter next to prefill, and can it be parallelised?
CPU only, runs on a laptop.

For several text sizes it times (1) one single-threaded encode call, (2) the same text split at safe boundaries into
chunks and encoded as a parallel batch, checking the token IDs are IDENTICAL, and (3) decoding. It compares the encode
time with a prefill-time estimate for the same number of tokens (Lab 04/09: ~50,000 prompt tokens/s on an H100).

    python tokenizer_bench.py --tokenizer Qwen/Qwen2.5-0.5B --sizes 100000 1000000 4000000
Needs: pip install transformers tokenizers (no torch required for tokenization).
Note: recent transformers versions load the Rust tokenizer even for use_fast=False, so no pure-Python baseline is timed.
"""
import argparse
import json
import os
import random
import time

from transformers import AutoTokenizer

WORDS = ("the model serves tokens from a cache while the scheduler batches requests and the kernel reads weights "
         "from memory so latency depends on bandwidth and queueing but throughput depends on batch size").split()


def make_text(n_chars, seed=0):
    rng = random.Random(seed)
    parts, size = [], 0
    while size < n_chars:
        w = rng.choice(WORDS)
        parts.append(w)
        size += len(w) + 1
    return " ".join(parts)[:n_chars]


def split_points(text, n_chunks):
    """Cut positions where the next character is a space and the previous one is not (so each chunk after the first
    starts with ' word', exactly how byte-level BPE pre-tokenizers would have split it anyway)."""
    cuts, target = [0], len(text) / n_chunks
    for i in range(1, n_chunks):
        j = int(i * target)
        while j < len(text) - 1 and not (text[j] == " " and text[j - 1] != " "):
            j += 1
        cuts.append(j)
    cuts.append(len(text))
    return cuts


def timed(fn, repeats=3):
    best, out = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--sizes", type=int, nargs="+", default=[100_000, 1_000_000, 4_000_000], help="characters")
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--prefill-tokens-per-s", type=float, default=50_000)
    ap.add_argument("--out", default="results/tokenizer_bench.json")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = []
    for n in a.sizes:
        text = make_text(n)
        t_one, ids = timed(lambda: tok(text, add_special_tokens=False)["input_ids"])
        cuts = split_points(text, a.chunks)
        pieces = [text[cuts[i]:cuts[i + 1]] for i in range(len(cuts) - 1)]
        t_par, parts = timed(lambda: tok(pieces, add_special_tokens=False)["input_ids"])
        merged = [t for p in parts for t in p]
        t_dec, _ = timed(lambda: tok.decode(ids))
        row = {"chars": n, "tokens": len(ids), "chars_per_token": round(n / len(ids), 2),
               "encode_single_s": round(t_one, 4), "encode_single_tok_per_s": round(len(ids) / t_one),
               "encode_parallel_s": round(t_par, 4), "parallel_speedup_x": round(t_one / t_par, 1),
               "parallel_ids_identical": merged == ids, "decode_s": round(t_dec, 4),
               "prefill_estimate_s": round(len(ids) / a.prefill_tokens_per_s, 3),
               "single_encode_pct_of_prefill": round(100 * t_one / (len(ids) / a.prefill_tokens_per_s), 1)}
        rows.append(row)
        print(json.dumps(row))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"tokenizer": a.tokenizer, "cpu_count": os.cpu_count(), "chunks": a.chunks, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()

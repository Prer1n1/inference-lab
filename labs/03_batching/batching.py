"""Lab 03: batching and context length in plain PyTorch (no serving engine).

For each (context, batch) it measures prefill time and decode throughput separately, plus peak memory.
Prefill is isolated by timing a 1-token generation; decode time is (N-token time) - (1-token time).

    python batching.py --model meta-llama/Llama-3.1-8B-Instruct --out results/batching.jsonl
Needs a CUDA GPU, PyTorch and transformers. Out-of-memory points are recorded, not fatal.
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def timed_generate(model, tok, ids, n_new):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n_new, min_new_tokens=n_new,
                   do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--contexts", type=int, nargs="+", default=[128, 1024, 4096])
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--out", default="results/batching.jsonl")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with torch.no_grad(), open(a.out, "a") as f:
        for ctx in a.contexts:
            for bs in a.batches:
                row = {"model": a.model, "context": ctx, "batch": bs, "new_tokens": a.new_tokens}
                try:
                    ids = torch.randint(1000, 20000, (bs, ctx), device="cuda")
                    timed_generate(model, tok, ids, 2)  # warm-up
                    torch.cuda.reset_peak_memory_stats()
                    t1 = timed_generate(model, tok, ids, 1)
                    tn = timed_generate(model, tok, ids, a.new_tokens)
                    decode_s = max(tn - t1, 1e-9)
                    row.update({"prefill_s": round(t1, 4), "prefill_tok_per_s": round(bs * ctx / t1),
                                "decode_ms_per_step": round(decode_s / (a.new_tokens - 1) * 1e3, 2),
                                "decode_tok_per_s": round(bs * (a.new_tokens - 1) / decode_s, 1),
                                "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
                except torch.cuda.OutOfMemoryError:
                    row["oom"] = True
                    torch.cuda.empty_cache()
                print(json.dumps(row))
                f.write(json.dumps(row) + "\n")
                f.flush()


if __name__ == "__main__":
    main()

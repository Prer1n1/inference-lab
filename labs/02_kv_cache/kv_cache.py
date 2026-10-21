"""Lab 02: the KV cache. Formula vs measurement, and what happens without it.

    python kv_cache.py --model meta-llama/Llama-3.1-8B-Instruct --out results/kv_cache.json
Needs a CUDA GPU, PyTorch and transformers.
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def kv_bytes_per_token(cfg, dtype_bytes=2):
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    kv_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    return 2 * cfg.num_hidden_layers * kv_heads * head_dim * dtype_bytes  # 2 = keys and values


def measure_cache_bytes(model, seq_len):
    """Memory held by the cache after prefilling seq_len tokens (base model only, so no huge logits tensor)."""
    ids = torch.randint(1000, 20000, (1, seq_len), device="cuda")
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    out = model.model(input_ids=ids, use_cache=True)
    del out.last_hidden_state
    torch.cuda.synchronize()
    held = torch.cuda.memory_allocated() - before
    del out, ids
    torch.cuda.empty_cache()
    return held


def gen_tokens_per_s(model, tok, prompt_len, new_tokens, use_cache):
    ids = torch.randint(1000, 20000, (1, prompt_len), device="cuda")
    kw = dict(max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False, use_cache=use_cache,
              pad_token_id=tok.eos_token_id)
    model.generate(ids, **{**kw, "max_new_tokens": 4, "min_new_tokens": 4})  # warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate(ids, **kw)
    torch.cuda.synchronize()
    return new_tokens / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[512, 2048, 8192])
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--out", default="results/kv_cache.json")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    per_token = kv_bytes_per_token(model.config)
    res = {"model": a.model, "formula_bytes_per_token": per_token,
           "formula_kib_per_token": per_token / 1024, "measured": []}
    with torch.no_grad():
        for L in a.seq_lens:
            held = measure_cache_bytes(model, L)
            res["measured"].append({"seq_len": L, "formula_mib": round(per_token * L / 2**20, 1),
                                    "measured_mib": round(held / 2**20, 1),
                                    "ratio_measured_over_formula": round(held / (per_token * L), 3)})
        with_cache = gen_tokens_per_s(model, tok, a.prompt_len, a.new_tokens, True)
        without = gen_tokens_per_s(model, tok, a.prompt_len, a.new_tokens, False)
    res["generation"] = {"prompt_len": a.prompt_len, "new_tokens": a.new_tokens,
                         "tok_per_s_with_cache": round(with_cache, 1), "tok_per_s_without_cache": round(without, 1),
                         "speedup_from_cache": round(with_cache / without, 2)}
    print(json.dumps(res, indent=2))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()

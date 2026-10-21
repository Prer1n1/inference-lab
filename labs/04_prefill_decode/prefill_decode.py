"""Lab 04: prefill vs decode, measured against the roofline from Lab 01.

Prefill: one forward pass over L prompt tokens.  Compared with the GPU's measured matmul ceiling.
Decode : one token per step with a KV cache, batch 1.  Compared with the memory-bandwidth floor
         (weights + KV cache must be read every step).
Note on FLOPs: 2 * parameters * tokens, ignoring attention FLOPs (small until contexts get long).
Eager PyTorch decode launches many tiny kernels, so small models can be limited by launch overhead
rather than bandwidth. That gap is the motivation for Lab 06 (CUDA graphs).

    python prefill_decode.py --gpu-info ../01_gpu_architecture/results/gpu_info.json --out results/prefill_decode.json
Needs a CUDA GPU, PyTorch and transformers.
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM


def cuda_seconds(fn, iters):
    fn()  # warm-up
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters / 1e3


def kv_bytes_per_token(cfg):
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    kv_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    return 2 * cfg.num_hidden_layers * kv_heads * head_dim * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--gpu-info", default=None, help="JSON from Lab 01 for roofline comparison")
    ap.add_argument("--out", default="results/prefill_decode.json")
    a = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    n_params = sum(p.numel() for p in model.parameters())
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    per_tok = kv_bytes_per_token(model.config)
    roof = json.load(open(a.gpu_info)) if a.gpu_info else {}
    bw = roof.get("measured_bandwidth_gb_s")
    peak_tf = max(roof["measured_matmul_tflops"].values()) if roof else None

    rows = []
    with torch.no_grad():
        for L in a.lengths:
            ids = torch.randint(1000, 20000, (1, L), device="cuda")

            def prefill():
                h = model.model(input_ids=ids, use_cache=True).last_hidden_state[:, -1:]
                return model.lm_head(h)
            t_pre = cuda_seconds(prefill, iters=5)

            out = model.model(input_ids=ids, use_cache=True)
            past = out.past_key_values
            nxt = model.lm_head(out.last_hidden_state[:, -1:]).argmax(-1)
            del out

            def decode_loop():
                nonlocal past, nxt
                for _ in range(a.decode_steps):
                    o = model.model(input_ids=nxt, past_key_values=past, use_cache=True)
                    past = o.past_key_values
                    nxt = model.lm_head(o.last_hidden_state[:, -1:]).argmax(-1)
            t_dec = cuda_seconds(decode_loop, iters=1) / a.decode_steps
            step_bytes = weight_bytes + per_tok * L
            row = {"prompt_len": L, "prefill_ms": round(t_pre * 1e3, 1), "prefill_tok_per_s": round(L / t_pre),
                   "prefill_tflops": round(2 * n_params * L / t_pre / 1e12, 1),
                   "decode_ms_per_step": round(t_dec * 1e3, 2),
                   "decode_achieved_gb_s": round(step_bytes / t_dec / 1e9, 1)}
            if bw:
                row["decode_floor_ms"] = round(step_bytes / (bw * 1e9) * 1e3, 2)
                row["decode_pct_of_bandwidth"] = round(100 * step_bytes / t_dec / 1e9 / bw, 1)
            if peak_tf:
                row["prefill_pct_of_peak_tflops"] = round(100 * row["prefill_tflops"] / peak_tf, 1)
            rows.append(row)
            print(json.dumps(row))
            del past
            torch.cuda.empty_cache()

    res = {"model": a.model, "params_b": round(n_params / 1e9, 2), "weights_gib": round(weight_bytes / 2**30, 2),
           "kv_kib_per_token": per_tok / 1024, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()

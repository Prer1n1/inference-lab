"""Lab 12, Part A: mixture-of-experts arithmetic from a model config. Laptop only, downloads the config (a few KB).

Computes total vs active parameters and, for a decode step at batch size B, how many experts are touched and therefore
how many weight bytes must be read, giving a bandwidth-floor on step time and a token-rate ceiling.

    python moe_math.py Qwen/Qwen1.5-MoE-A2.7B mistralai/Mixtral-8x7B-v0.1 --bandwidth-gb-s 3350

Assumption (stated, not measured): each token picks its top-k experts independently and uniformly. Real routers are
skewed, so real traffic touches FEWER distinct experts than this predicts. Supports Mixtral- and Qwen2-MoE-style configs.
"""
import argparse

from transformers import AutoConfig


def expert_geometry(c):
    """Return (n_experts, top_k, expert_params_each, shared_params_per_layer) for a supported config."""
    h = c.hidden_size
    if hasattr(c, "num_local_experts"):                         # Mixtral
        return c.num_local_experts, c.num_experts_per_tok, 3 * h * c.intermediate_size, 0
    if hasattr(c, "num_experts") and hasattr(c, "moe_intermediate_size"):   # Qwen2-MoE style
        shared = 3 * h * getattr(c, "shared_expert_intermediate_size", 0)
        return c.num_experts, c.num_experts_per_tok, 3 * h * c.moe_intermediate_size, shared
    raise SystemExit(f"unsupported MoE config: {c.model_type}")


def analyse(name, bandwidth_gb_s, bytes_per_param):
    c = AutoConfig.from_pretrained(name)
    c = getattr(c, "text_config", c)
    h, L = c.hidden_size, c.num_hidden_layers
    n_heads = c.num_attention_heads
    kv = getattr(c, "num_key_value_heads", n_heads)
    hd = getattr(c, "head_dim", None) or h // n_heads
    E, k, per_expert, shared = expert_geometry(c)
    attn = L * (h * n_heads * hd * 2 + h * kv * hd * 2)          # q, o and k, v projections
    router = L * h * E
    emb = c.vocab_size * h * (1 if getattr(c, "tie_word_embeddings", False) else 2)
    non_expert = attn + router + emb + L * shared
    total = non_expert + L * E * per_expert
    active = non_expert + L * k * per_expert
    print(f"\n{name}: layers {L}, hidden {h}, experts {E}, top-{k}, shared-expert params/layer {shared/1e6:.1f}M")
    print(f"  total params {total/1e9:.2f}B   active params/token {active/1e9:.2f}B   active share {100*active/total:.1f}%")
    print(f"  weights in BF16: {total*bytes_per_param/2**30:.1f} GiB (all experts must be resident)")
    print(f"  {'batch':>5} {'experts touched/layer':>22} {'bytes read/step (GB)':>21} {'floor ms':>9} {'ceiling tok/s':>14}")
    for B in (1, 2, 4, 8, 16, 32, 64, 128):
        touched = E * (1 - (1 - k / E) ** B)
        gb = (non_expert + L * touched * per_expert) * bytes_per_param / 1e9
        floor_ms = gb / bandwidth_gb_s * 1000
        print(f"  {B:>5} {touched:>14.1f} of {E:<5} {gb:>21.1f} {floor_ms:>9.2f} {B / floor_ms * 1000:>14.0f}")
    dense_ms = total * bytes_per_param / 1e9 / bandwidth_gb_s * 1000
    print(f"  (a dense model of the same TOTAL size would have a floor of {dense_ms:.2f} ms at every batch size)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--bandwidth-gb-s", type=float, default=3350)
    ap.add_argument("--bytes-per-param", type=float, default=2)
    a = ap.parse_args()
    for m in a.models:
        analyse(m, a.bandwidth_gb_s, a.bytes_per_param)

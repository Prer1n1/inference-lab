# Lab 02: the KV cache

**Status:** Scripts ready (not yet run)  
**Baseten posting items:** "KV-cache management", "KV-cache reuse"  
**Needs:** CUDA GPU with room for the model (16 GB for Llama-3.1-8B BF16 + a few GB). ~20 minutes. · **Depends on:** Lab 01

## 1. The question
How much memory does each token of context cost, and what does the cache buy us in speed?

## 2. Mental model
- Attention needs the keys and values of every earlier token. Caching them turns each decode step from "reprocess the whole sequence" into "process one token and read the cache".
- Cost per token = `2 (K and V) × layers × kv_heads × head_dim × bytes`. Llama-3.1-8B: 2 × 32 × 8 × 128 × 2 = 131,072 bytes = **128 KiB**. (Grouped-query attention is why `kv_heads` is 8, not 32.)
- Capacity: `(memory budget − weights − activations) ÷ KiB-per-token` = how many tokens of context fit across all users. In Lab 05 vLLM reported 438,000 tokens, which this formula reproduces from its 53.47 GiB.
- Without the cache, generating N tokens costs roughly N full forward passes over a growing context.

## 3. Predictions (fill in before running)
| Quantity | Prediction | Real | Verdict |
|---|---|---|---|
| KiB per token (formula, Llama-3.1-8B) | 128 | | |
| Measured cache MiB for 8,192 tokens ÷ formula | ~1.0 (1,024 MiB) | | |
| Tokens/s with cache vs without (512-token prompt, 128 new) | cache several times faster | | |
| Max tokens of cache in 50 GiB | 50 GiB ÷ 128 KiB ≈ 400k | | |

## 4. Experiment
```bash
cd labs/02_kv_cache
python kv_cache.py --model meta-llama/Llama-3.1-8B-Instruct --out results/kv_cache.json
```
The script reads the model config to compute the formula, prefills 512 / 2,048 / 8,192 tokens through the base model
(no logits, so no giant tensor pollutes the memory reading) and compares held memory with the formula, then generates
128 tokens with and without `use_cache`.

## 5. Results
_To fill in._

## 6. Interpretation
_To fill in: does the measured size match the formula? Does the cache speedup grow with context length? Why?_

## 7. Production takeaway
_To fill in. Hint: KV capacity, not compute, usually decides how many users fit. Which levers raise it (FP8 cache, GQA, paging, prefix reuse)?_

## 8. Pitfalls found
_To fill in._

## 9. Interview questions
1. Derive the KV-cache size per token for a model from its config. What does grouped-query attention change?
2. A GPU has 80 GB, the model is 16 GB. Roughly how many 4k-token conversations fit?
3. Why does PagedAttention help even when total memory is the same?

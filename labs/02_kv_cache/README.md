# Lab 02: the KV cache

**Status:** Scripts ready (not yet run)  
**Baseten posting items:** "KV-cache management", "KV-cache reuse"  
**Needs:** CUDA GPU with room for the model (16 GB for Llama-3.1-8B in BF16, plus a few GB). About 20 minutes. · **Depends on:** Lab 01

## 1. The question
How much memory does each token of context cost, how does that limit how many users fit, and what does the cache buy in speed?

## 2. Concepts

### 2.1 Attention in one paragraph
Each transformer layer turns every token into three vectors: a **query (Q)**, a **key (K)** and a **value (V)**. To produce the next token, the new token's query is compared with the keys of *all earlier tokens* to get weights, and those weights mix the earlier tokens' values. Keys and values of past tokens never change once computed.

### 2.2 Why cache K and V
Without a cache, generating token `n` would recompute K and V for all `n-1` earlier tokens in every layer: total work grows with the *square* of the length. With a **KV cache** you compute K and V once per token, store them, and each decode step processes one new token and *reads* the stored ones. The price is memory, and reading that memory every step.

### 2.3 How big is it
```
bytes per token = 2 (K and V) × layers × kv_heads × head_dim × bytes_per_element
```
| Model | Layers | KV heads | Head dim | BF16 bytes/token |
|---|---|---|---|---|
| Llama-3.1-8B | 32 | 8 | 128 | 2·32·8·128·2 = 131,072 = **128 KiB** |
| Llama-3.1-70B | 80 | 8 | 128 | 2·80·8·128·2 = 327,680 = **320 KiB** |

- An 8,192-token sequence on the 8B model holds **1 GiB** of cache. Fifty such sequences hold 50 GiB.
- **Capacity** = `(memory budget − weights − working memory) ÷ bytes per token`. In Lab 05, vLLM's log said 53.47 GiB → 438,000 tokens; the formula reproduces it exactly.

### 2.4 MHA, GQA, MQA: why `kv_heads` is 8, not 32
Classic **multi-head attention (MHA)** gives every query head its own K and V head. **Grouped-query attention (GQA)** shares each K/V head among a group of query heads; **multi-query attention (MQA)** shares one for all. Llama-3 has 32 query heads but only 8 KV heads, a 4× smaller cache than MHA. Model designers choose GQA precisely to cut KV memory and bandwidth.

### 2.5 The cache is also a bandwidth cost
Each decode step reads the weights *and* every sequence's cache. For the 8B model at 8k context, one sequence's cache (1 GiB) is about 7% of the weight bytes. At batch 32 with 4k contexts the caches total 16 GiB, *more than the weights*. Attention reading the cache is why long contexts and big batches slow each step (Lab 03).

### 2.6 Managing the cache
- **Naive allocation** reserves space for the maximum possible length per request. The vLLM paper found existing systems used only roughly 20-40% of reserved KV memory for real tokens; the rest was fragmentation and over-reservation.
- **PagedAttention** (vLLM): split each sequence's cache into fixed-size **blocks** (for example 16 tokens) allocated on demand and tracked in a per-sequence **block table**, like virtual-memory pages. No contiguous space is needed and waste is limited to the last partial block.
- **Prefix caching**: if a new request begins with tokens whose blocks already exist (identical shared prefix), reuse them and skip that prefill. Blocks are identified by a hash of their tokens and the tokens before them; only *full* blocks can be shared; unused blocks are evicted least-recently-used. (Lab 05's contaminated run was this feature working as designed on repeated prompts. Lab 08 measures it deliberately.)
- **KV-cache quantization**: storing the cache in FP8 halves its size (twice the tokens) at a small quality cost (Lab 07).
- **Eviction and preemption**: when blocks run out, the scheduler must pause or evict a sequence and later recompute or swap its cache (Lab 08).

## 3. Predictions (fill in BEFORE running)
| Quantity | Prediction | Real | Verdict |
|---|---|---|---|
| KiB per token for the model you load (from its config) | | | |
| Measured cache MiB for 8,192 tokens ÷ the formula | | | |
| Tokens/s with the cache vs without (512-token prompt, 128 new tokens): the ratio | | | |
| Cache tokens that fit in 50 GiB | | | |

## 4. Experiment, step by step
```bash
cd /root/inference-lab/labs/02_kv_cache
python kv_cache.py --model meta-llama/Llama-3.1-8B-Instruct --out results/kv_cache.json
```
**What the script does and why:**
1. Loads the model in BF16 and reads its config. *To compute the formula from first principles.*
2. For 512, 2,048 and 8,192 tokens, runs a prefill through the base model only (no output layer, so no giant logits tensor pollutes the reading) and measures the memory the cache holds. *Tests the formula against reality.*
3. Generates 128 tokens with `use_cache=True` and `use_cache=False` and compares tokens/s. *Shows what the cache buys.*

## 5. Results
_To fill in._

## 6. Interpretation
_To fill in: does the measured size match the formula, and by how much does it differ (padding, allocator rounding)? Does the speedup from the cache grow with prompt length? Why?_

## 7. Production takeaway
_To fill in._ Hint: KV capacity, not compute, usually decides how many users fit on a GPU. Which levers raise it (FP8 cache, GQA, paging, prefix reuse, shorter max length)?

## 8. Pitfalls to watch
- The Hugging Face cache object's internal layout changes between versions; the script measures memory held instead of inspecting tensors.
- Measuring after `generate` includes temporary buffers; measure right after the prefill.

## 9. Interview questions
1. Derive the KV-cache size per token from a model's config. What does grouped-query attention change?
2. A GPU has 80 GB and the model is 16 GB. Roughly how many 4k-token conversations fit?
3. Why does PagedAttention help even when total memory is unchanged?
4. Why can prefix caching only reuse full blocks, and what does that mean for prompt design?

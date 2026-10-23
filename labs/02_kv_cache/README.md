# Lab 02: the KV cache

**Status:** Done on an L40S (2026-10-23)  
**Baseten posting items:** "KV-cache management", "KV-cache reuse"  
**Needs:** CUDA GPU with room for the model (16 GB for Llama-3.1-8B in BF16, plus a few GB). About 20 minutes. · **Depends on:** Lab 01

## 1. The question
How much memory does each token of context cost, how does that limit how many users fit, and what does the cache buy in speed?

## 2. Mental model

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
| KiB per token for the model you load (from its config) | 2 × 32 × 8 × 128 × 2 bytes = 131,072 B = **128 KiB** (derived by hand in the session) | 128 KiB | Right |
| Measured cache MiB for 8,192 tokens ÷ the formula | about 1.0 (1,024 MiB); a 1,000-token sequence was worked out as 125 MiB = 0.122 GiB | **1,088 MiB, ratio 1.0625** | Within 6% |
| Tokens/s with the cache vs without (512-token prompt, 128 new tokens): the ratio | not recorded before the run | 39.2 vs 17.2 = **2.28×** | n/a |
| Cache tokens that fit in 50 GiB | 50 GiB ÷ 128 KiB = 409,600 tokens (arithmetic, not measured) | n/a | n/a |

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
**Environment:** 1× L40S (44.39 GiB), PyTorch 2.8.0+cu128, transformers 5.20.2, Llama-3.1-8B-Instruct in BF16 loaded from a local folder. Raw output: `results/kv_cache.json`.

| Sequence length | Formula (128 KiB × tokens) | Measured | Measured ÷ formula |
|---|---|---|---|
| 512 | 64 MiB | 76.1 MiB | 1.189 |
| 2,048 | 256 MiB | 272.0 MiB | 1.0625 |
| 8,192 | 1,024 MiB | 1,088.0 MiB | 1.0625 |

Generation (512-token prompt, 128 new tokens, Hugging Face `generate`, batch 1):

| | tokens/s | ms per token |
|---|---|---|
| With KV cache | 39.2 | 25.5 |
| Without KV cache | 17.2 | 58.1 |
| Speedup | **2.28×** | |

Single run; not repeated.

## 6. Interpretation
- **The formula holds.** 128 KiB per token matches the measurement within about 6% at 2,048 and 8,192 tokens. The excess is exactly 6.25% at both, and 12 MiB at 512 tokens. I did not find the cause (allocator rounding or temporary buffers are candidates); it is small enough to cover with a safety margin when planning capacity.
- **With the cache, decode is almost bandwidth-limited.** 25.5 ms per token against the Lab 01 floor of 24.4 ms (weights only) is about 96%.
- **Without it, each step recomputes the whole context.** A rough check: about 576 tokens on average × 16 GFLOP = 9.2 TFLOP, which is about 47 ms at the measured 194 TFLOP/s. The measured 58 ms is the same order. This is an estimate, not a decomposition.
- **The speedup should grow with prompt length**, because the no-cache cost grows with context while the cached step stays near the weight-read floor. Only a 512-token prompt was tested, so that growth is a prediction, not a result.
- **Not shown here:** the capacity (how many users fit) is arithmetic, not a measurement: 8,192 tokens = 1 GiB, so 50 GiB holds about 50 such sequences.

## 7. Engineering takeaway
The KV cache trades memory for compute: 128 KiB per token buys a near-floor decode step. Because that memory is per user and per token, **KV capacity, not compute, usually decides how many users fit on a GPU** (Lab 03 ran out of memory at batch 32 × 4,096 tokens on this card). The levers that raise capacity: FP8 KV (2× tokens), GQA (4× fewer KV heads, already in Llama-3), paging (no over-reservation), prefix reuse (shared prompts stored once) and shorter maximum lengths.

## 8. Pitfalls to watch
- The Hugging Face cache object's internal layout changes between versions; the script measures memory held instead of inspecting tensors.
- Measuring after `generate` includes temporary buffers; measure right after the prefill.

## 9. Interview questions
1. Derive the KV-cache size per token from a model's config. What does grouped-query attention change?
2. A GPU has 80 GB and the model is 16 GB. Roughly how many 4k-token conversations fit?
3. Why does PagedAttention help even when total memory is unchanged?
4. Why can prefix caching only reuse full blocks, and what does that mean for prompt design?

### Worked answers
1. **Derivation.** Per token the model stores one key and one value per layer per KV head: `2 × layers × kv_heads × head_dim × bytes` = 2 × 32 × 8 × 128 × 2 = 131,072 B = 128 KiB for Llama-3.1-8B (the config gives `num_hidden_layers` 32, `num_key_value_heads` 8, `hidden_size` 4096 ÷ 32 heads = head_dim 128). GQA shares each K/V head among a group of query heads: 8 KV heads instead of 32 makes the cache 4× smaller (512 KiB → 128 KiB per token).
2. **How many 4k conversations on 80 GB.** vLLM reserves about 90% = 72 GB; minus 16 GB of weights and working memory leaves about 53 GiB, which in Lab 05 was 438,000 tokens. 438,000 ÷ 4,096 ≈ 107 conversations (plus a margin for block rounding and per-request overhead).
3. **PagedAttention.** Memory is allocated in small fixed blocks on demand and tracked in a block table, so there is no contiguous space and no over-reservation for the maximum length. The vLLM paper reported existing systems using only about 20-40% of reserved KV memory for real tokens, so the same GPU holds many more sequences.
4. **Full blocks only.** Blocks are identified by a hash of their tokens (and the tokens before them), so only a complete block has a stable identity; a partial block changes as tokens are added. For prompt design: put shared content (system prompt, documents) first and stable, and put the varying part last, so the shared prefix covers as many whole blocks as possible.

## Further reading
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023).
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023).
- Shazeer, "Fast Transformer Decoding: One Write-Head Is All You Need" (2019), multi-query attention.
- Kipply, "Transformer Inference Arithmetic" (blog).

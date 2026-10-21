# Lab 14: long context and tokenization

**Status:** Part A done (CPU tokenization, runs on a laptop). Part B (GPU long-context serving) planned.  
**Baseten posting items:** example initiative "Making Kimi K3 tokenization 18x faster for million-token agentic workloads"; "KV-cache management"; "chunked prefill"; "tail latency" for long prompts  
**Needs:** Part A: a laptop. Part B: 1× H100 (or a 48 GB GPU for shorter contexts), about 2 hours · **Depends on:** Labs 02, 04, 08

## 1. The question
When prompts grow to 100,000 or a million tokens, what breaks first: memory, compute, or the CPU work around the model? And what can be done about each?

## 2. Mental model

### 2.1 Memory: the KV cache grows linearly with context
`KV bytes = tokens × bytes per token` (Lab 02). For an 8B model at 128 KiB per token:

| Context | KV cache for one sequence |
|---|---|
| 8,192 tokens | 1 GiB |
| 131,072 tokens (128k) | **16 GiB** (about the size of the weights) |
| 1,000,000 tokens | **about 122 GiB**, more than a whole 80 GB GPU |
So one long request can occupy a GPU's entire KV budget; at 128k only about 3 fit in a 50 GiB cache.

### 2.2 Compute: attention cost grows with the square of the context
Linear layers cost `2 × params × tokens` FLOPs. Attention adds `≈ 2 · L² · d_model` per layer for causal attention (the score and value products, halved by the causal mask). For the 8B model (32 layers, d_model 4,096):

| Prompt length L | Linear FLOPs | Attention FLOPs | Prefill time at 500 TFLOP/s effective |
|---|---|---|---|
| 8,192 | 1.3e14 | 1.8e13 (14%) | about 0.3 s |
| 131,072 | 2.1e15 | **4.5e15** (2× the linear part) | about **13 s** |
| 1,000,000 | 1.6e16 | **2.6e17** (16× the linear part) | about **9 minutes** |
(Estimates from the formula, not measurements; real kernels differ.) Attention FLOPs equal the linear FLOPs where `L = params ÷ (d_model × layers) ≈ 8.03e9 ÷ (4096 × 32) ≈ 61,000` tokens; beyond that attention dominates prefill. This is why very long contexts need **sparse or sliding-window attention**, **context parallelism** (splitting one prompt's attention across GPUs, for example ring attention), or models with compressed KV (latent attention).

### 2.3 Decode at long context
Each step reads the weights **and** the whole KV cache. At 128k, KV (16 GiB) is as large as the weights (15 GiB): the memory floor roughly doubles from about 4.8 ms to about 9.9 ms per step for one user, and batching helps less because each sequence adds its own huge cache (Lab 03).

### 2.4 Techniques
| Technique | What it does | Cost / limit |
|---|---|---|
| Chunked prefill | splits a long prompt so other users' decode is not stalled (Lab 08) | long-prompt TTFT stays long |
| Prefix caching | reuse the KV of a document that many requests share; saves the whole prefill | the cache itself is huge (a 100k-token document is about 12 GiB), so few fit; needs tiering |
| KV offload / tiering | spill cold KV to CPU memory or disk and bring it back | transfer time vs recompute time |
| KV quantization (FP8) | halves the cache (Lab 07) | small quality cost |
| Sliding-window / sparse attention | attend to a subset of tokens | model-specific; may cost quality |
| Context parallelism | split one sequence's prefill across GPUs | needs fast interconnects (Lab 13) |
| RoPE scaling (for example YaRN) | lets a model handle longer contexts than it was trained on | quality varies; must match the checkpoint's recipe |

### 2.5 The CPU side: tokenization and detokenization
Before a GPU sees a prompt, the server turns text into token IDs, and for every streamed step it turns new token IDs back into text.
- **Tokenization is O(text) on the CPU, usually single-threaded per request.** A million-token prompt is megabytes of text; if tokenization takes seconds, those seconds are added to TTFT *before the GPU does anything* and the GPU can sit idle.
- **Detokenization runs every step for every active stream.** At hundreds of concurrent streams the CPU can become the ITL bottleneck even though the GPU is fast.
- **Fast tokenizers** (Rust) are much quicker than pure-Python ones, but a single call is still sequential. A long text can be **split at safe boundaries and encoded in parallel**, provided the pieces tokenize exactly as the whole would (byte-level BPE attaches a space to the *following* word, so cutting just before a space is safe; this must be verified by comparing token IDs).
- Other costs: chat-template rendering, JSON encoding of large request bodies, and the HTTP server itself.

## 3. Prediction (before running Part A)
1. A single-threaded fast tokenizer encodes on the order of 100,000-1,000,000 tokens/s.
2. For a very long prompt, single-threaded encoding is a non-trivial fraction (more than 5%) of prefill time.
3. Splitting at space boundaries and encoding in parallel gives a speedup well under the core count (memory-bound, thread overhead) and produces identical token IDs.

## 4. Experiments

### 4.1 Part A: tokenization on a laptop
```bash
python tokenizer_bench.py --tokenizer Qwen/Qwen2.5-0.5B --sizes 100000 1000000 4000000
```
Environment: Windows laptop CPU with 16 logical processors, `transformers` 5.20.2 and `tokenizers` 0.23.3, synthetic text drawn from a 40-word vocabulary (5.97 characters per token; real English is closer to 4, so absolute rates are indicative only). Prefill reference: 50,000 prompt tokens/s (Lab 09's fit for an H100).

| Text | Tokens | Single encode | Parallel (16 chunks) | Speedup | IDs identical | Decode | Single encode as % of estimated prefill |
|---|---|---|---|---|---|---|---|
| 4,000,000 chars | 669,999 | 2.05 s (326k tok/s) | 0.76 s | **2.7×** | **yes** | 0.22-0.25 s | **15%** (of 13.4 s) |
Reproduced in two runs (2.057 / 2.043 s single, 0.765 / 0.761 s parallel).

**Interpretation**
- Prediction 1 held (326k tokens/s). Prediction 2 held: tokenizing a 670k-token prompt costs about **2 seconds, 15% of an estimated 13 s prefill**, and it lands entirely before the first token.
- Parallel encoding cut this to 0.76 s with **identical token IDs**, but only 2.7× on 16 logical processors: far from linear, so the gain is real but bounded (shared memory bandwidth, the split and merge work, and logical vs physical cores).
- The 100,000- and 1,000,000-character rows were **noisy between runs** (single-encode times varied by 3-4×, for example 0.034 s vs 0.133 s at 100k characters), so they are not quoted. Repeat counts and quieter conditions are needed before using small sizes.

**Limits:** synthetic low-entropy text; one tokenizer; a laptop CPU (a server CPU differs); the prefill time is an estimate, not a measurement; tokenization cost depends heavily on the tokenizer and text. We did **not** reproduce the claim in the posting's example (an 18× speedup); that is a different tokenizer and workload and we make no claim about it.
**A bug caught on the way:** the first version of this script "compared" slow vs fast tokenizers and reported 1.0×. Recent `transformers` versions silently load the same Rust tokenizer for `use_fast=False`, so it compared a thing with itself. Checking the loaded class exposed it; the comparison was replaced with single vs parallel encode.

### 4.2 Part B: long-context serving on a GPU (planned)
1. Serve a long-context model (Lab 05 setup with a larger `--max-model-len`; confirm the model's supported length).
2. Sweep prompt length (8k, 32k, 64k, 128k) at 1 user: measure TTFT, compare with the Section 2.2 estimates, and plot TTFT vs length (expect super-linear).
3. Decode ITL vs context at 1 user, compare with Section 2.3.
4. A mixed workload (short chat plus occasional 64k prompts): p99 ITL for chat users with chunked prefill off and on (Lab 08).
5. Prefix cache with a shared 32k-token document and many questions: hit rate, TTFT, and how many documents fit in the cache.
6. Measure server CPU during a long-prompt request (is the API process the bottleneck before the GPU starts?), and the time between request arrival and the first GPU step.

## 5. Engineering takeaway
Long context fails in layers: **KV memory** first (a few requests fill a GPU), then **prefill compute** (attention grows quadratically), and all along the **CPU** adds unoptimized seconds in tokenization and per-step detokenization. Each layer has its own fix, so a long-context serving plan names which bottleneck binds at which length. Tokenization is easy to ignore in a GPU-centric view, yet at 670k tokens it costs 15% of prefill even before any GPU work.

## 6. Interview questions
1. How big is the KV cache for a 1M-token context on an 8B model, and what does that imply for serving?
2. Why does prefill time grow faster than linearly with prompt length?
3. Why can the CPU become the bottleneck for long prompts or many streams, and how would you tell?
4. How can a long text be tokenized in parallel without changing the result, and how do you verify it?
5. When is prefix caching of a long document worth it, and what limits it?

## Further reading
- FlashAttention (Dao et al., 2022) and FlashAttention-2: why attention is memory-bound and how tiling helps.
- Ring Attention (Liu et al., 2023): context parallelism for very long sequences.
- YaRN (Peng et al., 2023): extending RoPE-based models to longer contexts.
- Hugging Face `tokenizers` documentation on batch encoding and parallelism.

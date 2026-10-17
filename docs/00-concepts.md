# Concepts you need before touching a GPU

## 1. Two phases with opposite bottlenecks

Generating text is not one workload. It is two.

**Prefill**: the model reads the whole prompt in one pass. Thousands of tokens are processed in parallel, so the GPU
does large matrix-matrix multiplies. This is **compute-bound**. It determines **TTFT** (time to first token).

**Decode**: the model produces one token at a time. Each step must read *all the model weights* (and the KV cache)
from GPU memory to produce a single token per sequence. Matrix-vector work, little arithmetic per byte moved. This is
**memory-bandwidth-bound**. It determines **ITL** (inter-token latency).

Everything in inference optimization follows from this asymmetry.

## 2. Back-of-envelope: why decode is slow (do this by hand)

Llama-3.1-8B in FP16: 8e9 params x 2 bytes = **16 GB** of weights.
H100 memory bandwidth: about **3.35 TB/s**.

Minimum time per decode step at batch size 1 = 16 GB / 3.35 TB/s = **about 4.8 ms**, no matter how fast the compute is.
So one user sees at most about 200 tokens/s, and the GPU's compute units are mostly idle.

**Batching is the fix.** At batch 32 the same 16 GB is read once, but 32 tokens come out. Throughput rises almost 32x
while step time barely moves, until compute or KV-cache memory becomes the limit. This is why your mock's
throughput rose from c=1 to c=8.

Quantization attacks the same number: FP8 weights = 8 GB, so the memory-bound step time halves.

## 3. KV cache

Attention needs the keys and values of every previous token. Recomputing them each step would be quadratic, so they
are cached. Size per token = `2 x layers x kv_heads x head_dim x bytes`.
Llama-3.1-8B: 2 x 32 x 8 x 128 x 2 = **128 KB per token**. A 4k-token sequence = 512 MB. 100 concurrent such
sequences = 51 GB. **The KV cache, not compute, is usually what limits how many users fit on a GPU.**

Consequences you will implement or measure:
- **PagedAttention** (vLLM): store the cache in fixed-size blocks like OS virtual memory, avoiding fragmentation.
- **Prefix caching**: if two requests share a prompt prefix, reuse its KV blocks and skip that prefill.
- **KV quantization**: FP8 cache = twice the sequences.

## 4. Scheduling

**Static batching** waits for a batch to finish. Short requests wait on long ones.
**Continuous batching** (vLLM, SGLang): at every decode step, finished sequences leave and waiting ones join.
**Chunked prefill**: split a long prompt into chunks mixed into decode steps, so one huge prompt does not stall every
other user's ITL. Trades a slightly worse TTFT for steadier ITL.

## 5. Speculative decoding

Since decode is memory-bound, checking k tokens costs almost the same as producing 1. A small draft model proposes
k tokens cheaply; the big model verifies all k in one pass and keeps the longest correct prefix. Output is
identical to normal decoding. Speedup depends on the **acceptance rate**, which depends on how well the draft
matches the target on *your* traffic.

## 6. Metrics, defined precisely

| Metric | Meaning | Driven by |
|---|---|---|
| TTFT | request sent to first token | queueing + prefill |
| ITL / TPOT | gap between tokens | decode step time under load |
| E2E | whole request | TTFT + (tokens-1) x ITL |
| Throughput | output tokens/s across all users | batch size, memory bandwidth |
| Goodput | throughput of requests that met the SLO | what actually matters |

Always report **p50 and p99**, never only the mean; tail latency is what customers feel.

## 7. Benchmarking traps (you will be asked about these)

- **Closed vs open loop.** My harness is closed-loop (fixed concurrency). Real traffic is open-loop (arrivals at a
  rate, independent of completions). Open loop exposes queueing collapse that closed loop hides.
- **Warm-up.** First requests include CUDA graph capture, cache cold-starts. Discard them.
- **Prompt realism.** Random-word prompts defeat prefix caching and speculative decoding. Use real text for those tests.
- **`ignore_eos`.** Fixes output length so runs are comparable, but real outputs vary.
- **Clock resolution.** Phase 0 lesson: on Windows `asyncio.sleep` returned early and produced fake numbers. The
  harness looked fine and the data was wrong. Always sanity-check results against a number you can compute by hand.

## Reading, in this order
1. vLLM paper: "Efficient Memory Management for LLM Serving with PagedAttention"
2. Horace He: "Making Deep Learning Go Brrrr From First Principles"
3. Kipply: "Transformer Inference Arithmetic"
4. FlashAttention (v1) abstract + the IO-awareness argument
5. Speculative decoding: Leviathan et al., 2022

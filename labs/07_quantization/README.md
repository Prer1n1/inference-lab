# Lab 07: quantization, speed vs capacity vs quality

**Status:** Planned (flags and checkpoints confirmed against the installed vLLM version when we run it)  
**Baseten posting items:** "Quantization (FP8/FP4, AWQ, GPTQ)", "tokens per GPU-hour", "clear latency/throughput/cost tradeoffs"  
**Needs:** 1× H100 (FP8 needs Hopper or Ada), about 2 hours (~$8) · **Depends on:** Labs 01, 05

## 1. The question
What do FP8 weights, INT4 weight-only quantization (AWQ, GPTQ) and an FP8 KV cache buy in speed and capacity, and what do they cost in quality?

## 2. Mental model

### 2.1 What quantization is
Storing numbers in fewer bits. A weight `w` is mapped to a small integer or low-precision float `q` plus a **scale** `s`, with `w ≈ s·q`. Scales are usually kept per **group** (for example every 128 weights) so one outlier does not ruin a whole tensor's precision.

### 2.2 Formats
| Format | Bits | Notes |
|---|---|---|
| BF16 | 16 | baseline; wide range, low precision |
| FP8 (E4M3 / E5M2) | 8 | native tensor-core support on Hopper and Ada; E4M3 (more mantissa) for weights/activations, E5M2 (more range) for gradients |
| INT8 | 8 | integer; needs scales; common for weights and activations with care |
| INT4 | 4 | weight-only in practice (AWQ, GPTQ); needs group scales |
| FP4 (MXFP4 / NVFP4) | 4 | block-scaled float on newest hardware (Blackwell class) |

### 2.3 Why it speeds decode (and what it does not do)
- **Weight-only quantization (W4A16, W8A16)**: weights are stored small, read from HBM small, then converted to BF16 inside the kernel; the math is still BF16. Since decode is memory-bound (Lab 01), reading half or a quarter of the bytes cuts step time toward `bytes ÷ bandwidth`: for the 8B model, floors of about 4.8 ms (BF16), 2.4 ms (8-bit), 1.2 ms (4-bit). Prefill, which is compute-bound, gains little.
- **Weight + activation quantization (W8A8, FP8)**: weights *and* activations are low precision and the matmul runs on FP8 tensor cores, which on an H100 have about twice the BF16 throughput. This helps prefill too.
- **KV-cache quantization (FP8 KV)**: halves cache bytes: twice the tokens in the same memory (capacity) and less cache traffic per step.
- Real speedups fall short of the byte ratio because of dequantization cost, non-quantized layers (embeddings, norms, sometimes the output head), and fixed overheads.

### 2.4 How low-bit methods keep quality
- **Outliers**: a few channels have much larger magnitudes than the rest and dominate error. Methods handle them differently.
- **AWQ** (activation-aware weight quantization): finds the weight channels that matter most by looking at activation magnitudes on calibration data, and scales them to protect their precision before rounding.
- **GPTQ**: quantizes layer by layer using second-order (Hessian) information, rounding each weight and adjusting the not-yet-quantized ones to compensate for the error.
- **Calibration data**: methods that use it depend on it; it should resemble the target workload.
- **Quality measures**: perplexity on held-out text (fast, a blunt signal) and task accuracy on a small fixed eval (slower, closer to what customers notice). Always compare against the BF16 baseline with identical prompts and settings, and check where degradation hurts (long context, math, code, non-English).

### 2.5 FP4 and block-scaled formats (concept only in this plan)
4-bit floating-point formats (for example **MXFP4** and **NVFP4**) store 4-bit values with a **shared scale per small block** (16 or 32 values), which keeps outliers manageable better than a single tensor scale. The newest GPU generation (Blackwell class) has tensor cores that compute in these formats **natively**, so, unlike INT4 weight-only quantization (which dequantizes to BF16 before the math), FP4 can speed up **both** decode (a quarter of the BF16 bytes) and prefill (higher peak FLOP/s).
- Arithmetic to write down: an 8B model in FP4 is about 4 GB; the decode floor on a 3.35 TB/s GPU is about **1.2 ms**, the same bytes as INT4.
- **Not measurable on Hopper or Ada GPUs**, which cannot run FP4 natively. Unless Blackwell hardware is rented, this lab treats FP4 as a written comparison with INT4 (native compute vs dequantize; block scales vs group scales) and marks it clearly as **not measured**.
- Quality questions are the same as for any low-bit format: calibration data, outlier handling, and where degradation shows up.

### 2.6 Cost accounting
Use **goodput at a latency target**, not peak tokens/s: a config that is 30% faster at peak but breaches the latency target at your operating load is not cheaper.

## 3. Predictions (fill in BEFORE running)
| Quantity | Prediction (arithmetic from Lab 01 floors) | Real | Verdict |
|---|---|---|---|
| ITL at 1 user: FP8 vs BF16 | | | |
| ITL at 1 user: INT4 vs BF16 | | | |
| Prefill speedup (TTFT at a long prompt) from FP8 vs from INT4 | | | |
| KV capacity (tokens) with FP8 KV cache vs BF16 | | | |
| Perplexity change per config | | | |

## 4. Steps and why
1. **Baseline BF16** with the Lab 05 settings. *Everything is a delta against it.*
2. **FP8 weights** (online or a pre-quantized checkpoint, confirmed at run time). *The main modern option on Hopper.*
3. **AWQ INT4** and **GPTQ INT4** checkpoints. *Compare the two INT4 methods on speed and quality.*
4. **FP8 KV cache** alone and combined with the best weight format. *Isolates the capacity gain.*
5. **Speed:** ITL at 1 user (compare with the floors), throughput at 64 users, TTFT on long prompts. **Capacity:** KV-cache tokens in the server log.
6. **Quality:** perplexity plus a small task eval on a fixed subset for every config.
7. **Cost:** goodput at a fixed latency target → $/M tokens per config.

## 5. Results, 6. Interpretation, 7. Engineering takeaway, 8. Pitfalls
_To fill in._ Pitfalls: comparing different calibration data; changing several variables at once; quality evals too small to see a 1% change; pre-quantized checkpoints from different sources behaving differently; forgetting that a quantized model can change which kernels run (profile it).

## 9. Done when
- One table: config × (ITL, tokens/s at target, KV capacity, quality delta, $/M tokens).
- Predicted vs measured decode speedup from fewer bytes, with an explanation of the gap.
- A recommendation per workload type.
- A written FP4 comparison with INT4, clearly marked as not measured unless Blackwell hardware was used.

## 10. Interview questions
1. Why does halving weight bytes not halve end-to-end latency?
2. When is weight-only INT4 a poor choice compared with FP8 W8A8?
3. How do you decide a quality loss is acceptable for a customer?
4. How do AWQ and GPTQ differ in how they preserve accuracy?

## Further reading
- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (2023).
- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2022).
- Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" (2022).
- Micikevicius et al., "FP8 Formats for Deep Learning" (2022).
- Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (2022): the outlier problem.

# Lab 13: new model bring-up, time-boxed

**Status:** Planned  
**Baseten posting items:** "Help the team bring up and tune new model architectures on new hardware quickly, often in the same week they are released"  
**Needs:** 1× H100, about 3 hours (~$12), spread over a few days · **Depends on:** Labs 05-08

## 1. The question
How fast can I take a newly released architecture from "does not run" to "tuned and benchmarked", and what checklist makes the next one faster?

## 2. Concepts

### 2.1 What can be new about a model
| Component | Variants you will meet | Why it matters for serving |
|---|---|---|
| Attention | MHA, GQA, MQA, latent attention (MLA), sliding window, hybrid layers | KV-cache size and kernel support |
| Position encoding | RoPE with different scaling schemes for long context | numerics must match exactly or quality silently degrades |
| Feed-forward | dense MLP, **mixture of experts (MoE)** | memory and compute behavior (below) |
| Normalization / activation | RMSNorm variants, gated activations | fused-kernel coverage |
| Tokenizer and chat template | new vocabularies, special tokens | garbled output if wrong |
| Precision | native FP8/FP4 checkpoints | needs matching kernels and hardware |

### 2.2 Mixture of experts in serving terms
An MoE layer has many expert MLPs but routes each token to only a few (top-k). **Total parameters** set the memory footprint (all experts must be resident); **active parameters** set the per-token compute. Decode at small batch reads only the chosen experts, but as batch grows more distinct experts are touched each step, so bytes read per step rise toward the full model: the memory-bound picture changes with batch size. Large MoEs are typically served with **expert parallelism** across GPUs, adding all-to-all communication.

### 2.3 What "bring-up" consists of
1. **Load**: map checkpoint tensor names to the engine's module structure; handle sharding and dtypes.
2. **Make it correct**: implement or enable the architecture's components; compare against the reference implementation.
3. **Make it fast**: right kernels, quantization, parallelism, scheduler settings.
4. **Make it safe to ship**: benchmark, regression tests, documentation.

### 2.3.1 Verifying correctness
- Compare **logits** (or top-token agreement) against the reference framework on fixed prompts: with identical weights and greedy decoding, tokens should match for dozens of steps; small numerical differences grow, so compare early positions tightly.
- Test the **chat template** end to end; test long context and, if relevant, multimodal inputs.
- A model that "runs" but produces subtly wrong text is the dangerous failure.

### 2.4 New hardware
New GPU generations bring new precisions (FP8, FP4), new instructions, and a software stack (drivers, CUDA, compilers, libraries) that lags the silicon. Bring-up on new hardware often means finding which kernels are missing or slow and which library versions work together.

## 3. Predictions (fill in BEFORE starting)
| Quantity | Prediction | Real | Verdict |
|---|---|---|---|
| Hours to a correct first output | | | |
| Hours to a baseline benchmark | | | |
| Memory footprint and KV bytes/token (from the config, before running) | | | |
| Where the first failure will occur (loading, kernels, template, numerics) | | | |

## 4. Steps and why
1. **Pick a recent open model** with a feature the engine supports only in a recent or nightly build. *Realistic: the point is the unfamiliar.*
2. **Read the config first** and compute weights, KV bytes/token and expected decode floor on your GPU. *Predictions before running, as always.*
3. **Day 1: serve it and verify correctness** against the reference (logits or token match, chat template). Log every failure and its cause.
4. **Day 2: baseline benchmark** with the harness (Lab 05 methodology: fresh prompts, repeats, server-side load check).
5. **Day 3: tune** (quantization, parallelism, scheduler flags) and record gains.
6. **Second GPU type** (the "new hardware" analogue): rerun the benchmark; note what changed.
7. **Distill a checklist** of the steps that cost time, for the next model.

## 5. Results, 6. Interpretation, 7. Production takeaway, 8. Pitfalls
_To fill in._ Pitfalls: trusting "it generates fluent text" as proof of correctness; benchmarking before correctness is verified; unpinned library versions; tuning before a baseline exists.

## 9. Done when
- A timeline log with time spent per stage.
- A reusable bring-up checklist.
- A benchmark table on two GPU types.

## 10. Interview questions
1. What do you check first when a new model produces garbage in the engine?
2. How does an MoE model change the memory-vs-compute picture?
3. What would you automate to cut bring-up from days to hours?
4. How do you verify numerical correctness of a port without a ground-truth output?

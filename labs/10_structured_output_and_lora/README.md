# Lab 10: structured output and multi-LoRA serving

**Status:** Planned (flags confirmed against the installed vLLM version when we run it)  
**Baseten posting items:** "LoRA", "guided generation for structured outputs"  
**Needs:** an inexpensive GPU, about 1 hour (~$3) · **Depends on:** Lab 05

## 1. The question
What do JSON-schema guided decoding and serving many LoRA adapters cost in latency and throughput?

## 2. Concepts

### 2.1 Guided (constrained) decoding
Goal: the model's output must follow a grammar, such as valid JSON matching a schema, a regex, or a choice among options.
Mechanism: compile the constraint into a **finite-state machine** or pushdown automaton over the *token vocabulary*. At each step the automaton, given the tokens generated so far, says which tokens are allowed next. The engine sets the logits of all disallowed tokens to `−∞` before sampling, so the model *cannot* produce an invalid continuation.
- Guarantees **syntax**, not **meaning**: the JSON is valid, the values may still be wrong.
- Costs: **compile time** for a new schema (can add to TTFT; caches help), and **per-step mask computation** over a vocabulary of 100k+ tokens (done on CPU or optimized in libraries such as xgrammar/outlines-style backends), which can slow ITL when not overlapped with GPU work.
- Quality interaction: forcing structure can change what the model would have said; evaluate task accuracy, not only validity.

### 2.2 LoRA
Fine-tuning every weight is expensive. **Low-rank adaptation** freezes the base weight `W` and learns a small update `ΔW = (α/r)·B·A` where `A` is `r × K` and `B` is `N × r` with small rank `r` (for example 8-64). The adapter has `r(K + N)` parameters per adapted matrix instead of `K·N`: tens of MB for a whole model, against 16 GB for the base.
- Inference can **merge** (`W' = W + ΔW`) for a single adapter at zero extra cost, but then the replica serves only that adapter.
- Or keep adapters **separate**: compute `y = Wx + (α/r)·B(Ax)`; the extra work is two skinny matmuls per adapted layer.

### 2.3 Multi-LoRA serving
Many customers each have an adapter; one base model in memory serves them all. Requests for different adapters are batched together: the base matmul is shared by the whole batch and the low-rank part is computed per request using **batched gather kernels** (SGMV/BGMV style, from systems like Punica and S-LoRA). Memory: adapters live in GPU memory up to a cap (`max_loras`) and are loaded/evicted on demand. Costs: extra kernels per layer, adapter loading latency on a miss, and less kernel efficiency when the batch mixes many adapters.

## 3. Predictions (fill in BEFORE running)
| Quantity | Prediction | Real | Verdict |
|---|---|---|---|
| ITL overhead of JSON-schema decoding | | | |
| TTFT overhead for a first-time schema vs a repeated one | | | |
| Share of valid outputs: unconstrained vs constrained | | | |
| Throughput with 1 vs 4 vs 16 adapters in mixed traffic, relative to base | | | |

## 4. Steps and why
1. **Structured output**: same prompts, unconstrained vs JSON-schema-constrained. Measure TTFT, ITL, throughput and the **validity rate** (parse every output). *Quantifies the cost of the guarantee and the failure rate without it.*
2. **Schema reuse**: first request with a new schema vs repeats. *Separates compile cost from per-token cost.*
3. **LoRA**: serve the base with 1, 4 and 16 adapters (rank 16), send traffic that mixes adapters evenly, compare with base-only. *Finds the overhead and the point where it hurts.*
4. **Adapter churn**: more adapters than `max_loras` to force eviction. *Shows the miss penalty.*

## 5. Results, 6. Interpretation, 7. Production takeaway, 8. Pitfalls
_To fill in._ Pitfalls: testing only one trivial schema; ignoring prompt design (the model should be told the format too, or quality suffers); using identical adapters (no realistic mixing); rank and target modules change cost.

## 9. Done when
- An overhead table for guided decoding and for N adapters.
- A recommendation on caps and on when to merge an adapter instead of serving it separately.

## 10. Interview questions
1. How does constrained decoding mask tokens and where does its cost come from?
2. Why can many LoRA adapters share one base model's batch?
3. What breaks first as the adapter count grows?
4. When would you merge a LoRA into the base weights?

# Lab 15: engine comparison, vLLM vs SGLang vs TensorRT-LLM

**Status:** Planned (harness and matrix runner ready; engine installs and flags confirmed when we run it)  
**Baseten posting items:** "Strong familiarity with ML libraries, especially PyTorch, TensorRT, or TensorRT-LLM"; "Familiarity with inference engines such as vLLM, SGLang or TensorRT-LLM"; "Contribute upstream to open-source inference engines (vLLM, SGLang, TensorRT-LLM)"; "benchmarking frameworks across model architectures"  
**Needs:** 1× H100, about 3-4 hours (~$12-16); TensorRT-LLM setup is the long pole · **Depends on:** Labs 05, 06, 08

## 1. The question
For the same model, hardware and workload, how do the three main open inference engines differ in throughput, latency, features and operational cost, and what explains the differences?

## 2. Mental model

### 2.1 What each engine is built around
| Engine | Core ideas | Typical strengths | Typical costs |
|---|---|---|---|
| **vLLM** | PagedAttention block KV memory, continuous batching, a large community and model coverage, OpenAI-compatible server | breadth of models and features, fast adoption of new models, easy to start | defaults that need tuning; features vary by version |
| **SGLang** | RadixAttention (a radix tree over cached prefixes for automatic reuse), a low-overhead scheduler, strong structured-output support, a programming front end | prefix-heavy and multi-turn workloads, agent loops, constrained decoding | smaller model coverage at times; fast-moving flags |
| **TensorRT-LLM** | NVIDIA's engine: model compiled into optimized TensorRT kernels and plugins, in-flight batching, paged KV, aggressive FP8/FP4 support on NVIDIA hardware | peak performance on NVIDIA GPUs, first access to new numeric formats | build step per model and configuration, more rigid, steeper learning curve, NVIDIA-only |
(Descriptions are from the projects' own designs; versions change fast. Confirm each claim against the versions you install before quoting it.)

### 2.2 Why comparisons are easy to get wrong
- **Defaults differ**: maximum batch tokens, chunked prefill, CUDA graphs, prefix caching on or off, scheduler policy, quantization defaults. Comparing defaults compares *configurations*, not *engines*.
- **Effort differs**: an engine tuned by its author vs one run with defaults. Give each equal tuning effort and **write down every flag**.
- **Workload decides the winner**: a shared-prefix workload favors radix-style caching; a unique-prompt workload does not; long outputs favor decode efficiency; short outputs favor prefill and scheduling overhead.
- **Measurement differs**: client behavior, streaming granularity, token counting. Use **one client** (our harness), server-reported token counts and the server-side load check.
- **Engine versions move monthly**: a result is a snapshot. Record versions and dates.
- **Cost is not only speed**: build time, startup time, memory headroom, stability, observability, API compatibility, model coverage and how fast a new model arrives.

### 2.3 A fair protocol
1. Same model checkpoint and precision (BF16 first; then FP8 where all three support it), same `max_model_len`, same GPU, same driver and CUDA.
2. Each engine in its own clean environment or container.
3. Each engine tuned within a stated budget (for example the documented recommended flags plus a small sweep of batch-token and sequence limits).
4. Same workloads: (a) Lab 05's chat shape; (b) long prompt, short output; (c) short prompt, long output; (d) a shared 2,000-token prefix; (e) structured-output requests; run closed loop (1, 16, 64, 128 users) and open loop at fixed rates.
5. Fresh prompts except where (d) is the point; warm-up; at least two repeats; verify the server saw the load asked for.
6. Report p50/p99 TTFT and TPOT, tokens/s, **goodput at a latency target**, startup and build time, and what had to be tuned.

## 3. Predictions (write BEFORE running)
| Quantity | Prediction | Real | Verdict |
|---|---|---|---|
| Throughput ranking at 64 users, unique prompts, BF16 | | | |
| TTFT at 1 user | | | |
| Winner on the shared-prefix workload, and by how much | | | |
| Startup time (including any build step) per engine | | | |
| What changes when FP8 is enabled in each | | | |

## 4. Experiments, step by step
1. **Install each engine** separately and record versions (`pip`, container tags). *Reproducibility.*
2. **Serve the same model** in each, with the harness-compatible OpenAI endpoint. *One client for all.*
3. **Write a matrix config** (`bench/run_matrix.py`): three servers by workloads. *The runner starts each server, runs every workload and stops it, so the comparison is mechanical.*
4. **Run closed- and open-loop sweeps** with warm-up and repeats; collect everything into one JSONL file.
5. **Tune fairly**: for each engine, try its documented recommended settings and a small sweep of the batch-token and sequence-limit knobs; keep the best that preserves the latency target.
6. **Profile the largest gap** with the Lab 06 method: where does the faster engine spend less time (scheduling overhead, kernels, CPU work)?
7. **Operational comparison**: time to first successful request, size of the build step, metrics exposed, structured-output and LoRA support, new-model turnaround.
8. **Write it up** with every flag, version and caveat, and a clear statement of which workloads each engine wins.

## 5. Results, 6. Limits, 7. Engineering takeaway
_To fill in._ Known limits to state: one GPU type, one model family, a specific date and version, tuning effort bounded. Do not generalize "engine X is faster" beyond the workloads tested.

## 8. Done when
- A matrix table (engine × workload × load) with p50/p99, goodput and cost per million tokens.
- Every flag and version recorded; two repeats agreeing.
- A recommendation per workload type, with the one or two reasons found by profiling.
- At least one upstream issue or PR candidate from the comparison (Lab 21).

## 9. Interview questions
1. How would you design a fair benchmark between two inference engines?
2. Why can the same engine look 30% faster or slower depending on defaults?
3. When would you pick TensorRT-LLM over vLLM or SGLang, and what would you give up?
4. Why does a shared-prefix workload favor radix-tree caching, and how would you test that claim?
5. What would you check before trusting a blog post's "engine X is 2× faster than engine Y"?

## Further reading
- vLLM paper (Kwon et al., 2023); SGLang paper (Zheng et al., 2023) on RadixAttention.
- NVIDIA TensorRT-LLM documentation: in-flight batching, the build workflow and quantization support.
- Each project's own benchmark scripts and the issues discussing methodology.

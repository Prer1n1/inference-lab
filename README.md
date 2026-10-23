# inference-lab

A hands-on, measured path through LLM inference performance, from GPU fundamentals to production serving. It is built so that
every requirement in Baseten's *Inference Performance* engineering posting is both **understood (theory)** and **demonstrated
(practice)**. Each lab is a short lesson: a question, the mental model, a prediction made *before* measuring, the experiment with
reasons for every step, measured results, an interpretation that separates what was measured from what is inferred, stated limits,
an engineering takeaway, and interview questions. Mistakes found along the way are kept and explained.

**The guiding question:** given one GPU and a latency target, what is the cheapest way to serve an LLM, and why does each setting change the numbers?
Serving cost is `GPU $/hour ÷ tokens/hour`; every technique raises tokens/hour or lowers latency at the same tokens/hour, and they trade against each other.

**The long-term goal:** given a workload, a latency target and a budget, design a serving setup, benchmark it honestly, find its bottleneck
(memory, compute, communication, CPU or scheduling), tune it, and explain every trade-off.

## Learning path
Fundamentals: GPU architecture → KV cache → batching → prefill vs decode → a real serving engine.
Performance engineering: profiling → quantization → scheduling and caching → custom scheduling → speculative decoding → structured output and LoRA.
Scale and systems: mixture of experts → multi-GPU parallelism → long context and tokenization → engine comparison → routing and disaggregation.
Craft and production: kernels → new-model bring-up → Kubernetes, autoscaling, observability → capacity and cost → upstream contribution.

## Lab status
Status values: **Done** (run and written up), **Part A done** (laptop work finished, GPU work planned), **Scripts ready** (not yet run), **Planned**.

| Lab | Topic | Status | Key result so far |
|---|---|---|---|
| [00](labs/00_concepts) | Concepts primer | Done | |
| [01](labs/01_gpu_architecture) | GPU architecture and the roofline | Done (L40S) | measured 655 GB/s (76% of spec) and 196 TFLOP/s (54%); ridge point 299; 8B BF16 decode floor 24.4 ms |
| [02](labs/02_kv_cache) | KV cache | Done (L40S) | 128 KiB/token confirmed within 6%; KV cache makes decode 2.3× faster at a 512-token prompt, ~96% of the bandwidth floor |
| [03](labs/03_batching) | Batching and context length | Done (L40S) | batch 1 → 32 at short context: step +19%, throughput 27×; batch 32 × 4k context ran out of memory (overhead ~0.44 GiB per sequence beyond the KV formula) |
| [04](labs/04_prefill_decode) | Prefill vs decode against the roofline | Done (L40S) | prefill 82-100% of the matmul ceiling (99.9% at 2k tokens); decode 94-104% of the bandwidth floor |
| [05](labs/05_vllm_baseline) | vLLM baseline, concurrency sweep, cost per token | Measured (levels ≥128 need a re-run) | ~100 users: 44× throughput, $6.91 → ~$0.16 per M tokens; two benchmark bugs found and guarded |
| [06](labs/06_profiling) | Profiling: where a step's time goes | Planned | |
| [07](labs/07_quantization) | Quantization: FP8, AWQ, GPTQ, FP8 KV (FP4 as theory) | Planned | |
| [08](labs/08_scheduling_and_caching) | Chunked prefill, token budget, prefix caching | Planned | |
| [09](labs/09_custom_scheduling) | Custom scheduling: simulator and real scheduler | Part A done | simulator within 2-8% of Lab 05 on held-out points; policy only matters once a queue forms (14× chat TTFT difference at overload) |
| [10](labs/10_speculative_decoding) | Speculative decoding | Planned | |
| [11](labs/11_structured_output_and_lora) | Guided decoding and multi-LoRA | Planned | |
| [12](labs/12_moe_sparse_models) | Mixture of experts | Part A done | MoE decode is ~5× cheaper in bytes at batch 1; the advantage vanishes by batch 32-64 |
| [13](labs/13_multi_gpu_inference) | Multi-GPU: tensor, pipeline, expert parallelism | Scripts ready | |
| [14](labs/14_long_context_and_tokenization) | Long context and tokenization | Part A done | tokenizing 670k tokens ≈ 2 s (15% of est. prefill); parallel encode 2.7×, identical IDs |
| [15](labs/15_engine_comparison) | vLLM vs SGLang vs TensorRT-LLM | Planned | |
| [16](labs/16_routing_and_disaggregation) | Cache-aware routing, prefill/decode disaggregation | Planned | |
| [17](labs/17_kernels_triton_cuda) | Kernels: Triton, CUDA C++, CUTLASS | Planned | |
| [18](labs/18_new_model_bringup) | New-model bring-up | Planned | |
| [19](labs/19_production_serving) | Kubernetes, autoscaling, observability, multi-region design | Planned | |
| [20](labs/20_capacity_and_cost_report) | Capacity and cost report | Planned | |
| [21](labs/21_upstream_contribution) | Upstream contribution | Planned | |

## What the job asks for, and where this repo answers it

| Posting item | Lab(s) |
|---|---|
| Deep understanding of GPU architecture; reason from first principles about time and memory | 01, 04, 06 |
| KV-cache management and reuse | 02, 08, 14 |
| Continuous batching and LLM optimization basics | 03, 05 |
| Quantization (FP8/FP4, AWQ, GPTQ) | 07 (FP4 as theory unless Blackwell hardware is used) |
| Chunked prefill | 04, 08 |
| Custom scheduling algorithms; working in runtime internals | 08, 09 |
| Speculative decoding | 10 |
| LoRA; guided generation for structured outputs | 11 |
| New (including MoE) architectures, brought up quickly | 12, 18 |
| Multi-GPU and multi-node serving | 13 |
| Long-context workloads; tokenization performance | 14 |
| PyTorch, TensorRT-LLM, vLLM, SGLang familiarity | 01-04 (PyTorch), 05 (vLLM), 15 (all engines) |
| Profile end to end: kernel launch overhead, memory layout, scheduling; cross-layer tail-latency investigations | 06, 09 |
| Cache-aware routing; prefill/decode disaggregation; load balancing | 16 |
| GPU kernels (CUDA, Triton, CUTLASS) | 17 |
| Build benchmarking frameworks (models, batch sizes, sequence lengths, hardware) | `bench/` (+ `run_matrix.py`), 05, 15 |
| Distributed serving: autoscaling, observability, incident response, multi-region/multi-cloud | 19 |
| Tokens per GPU-hour, utilization, latency/throughput/cost tradeoffs | 05, 07, 20 |
| Contribute upstream to vLLM / SGLang / TensorRT-LLM | 21 |

What the labs cannot cover: people management and hiring (the Engineering Manager posting), and anything needing hardware that is not
rented (for example FP4 needs a Blackwell-class GPU). The labs say so where it applies.

## Where to start
1. [docs/study-plan.md](docs/study-plan.md): the order of work, costs and the rules that keep results credible.
2. [labs/00_concepts](labs/00_concepts), then the *Mental model* of each lab. Terms are defined in [docs/glossary.md](docs/glossary.md).
3. [docs/session-a-labs-01-04.md](docs/session-a-labs-01-04.md): the exact commands for the next GPU session.

## Layout
```
bench/            load generator (closed/open loop, goodput, GPU telemetry), mock server, matrix runner, plotting
tests/            self-tests (harness, simulator) that run without a GPU
labs/NN_topic/    README (the lesson), scripts, results/ (raw data and outputs)
templates/        the lab template every README follows
docs/             study plan, GPU-session runbooks, pod workflow, glossary
```

## Quick start (laptop, no GPU)
```
python -m venv .venv && .venv\Scripts\pip install -r requirements.txt      # Linux/macOS: .venv/bin/pip
.venv\Scripts\python tests\test_harness.py                                  # benchmark tool vs the mock server
.venv\Scripts\python tests\test_simulator.py                                # scheduler simulator checks
.venv\Scripts\python labs\09_custom_scheduling\simulator.py validate        # simulator vs measured Lab 05 numbers
.venv\Scripts\python labs\12_moe_sparse_models\moe_math.py Qwen/Qwen1.5-MoE-A2.7B   # needs: pip install transformers
```
Against a real server, point `bench/bench.py --url ... --model ...` at it; `python bench/bench.py --help` lists load models, SLO goodput,
shared-prefix and output-length options. `bench/run_matrix.py` runs a server × workload matrix from one config file.

## Running on a GPU
Follow [docs/pod-workflow.md](docs/pod-workflow.md). Labs 01-04 use `requirements-labs.txt`; vLLM labs use a separate environment (`requirements-vllm.txt`).
GitHub is the source of truth: edit locally, push, pull on the pod, run, copy results back, **terminate the pod**.

## Conventions (each exists because it cost a measurement)
1. Write predictions, with the arithmetic, before running. Score them afterwards.
2. Fresh random prompts per level unless a lab is deliberately testing caching.
3. Warm up, then repeat. Two runs that disagree mean something is wrong; find out what before reporting either.
4. Report p50 and p99, and ranges when repeats disagree.
5. Verify that the load generator delivered the load you asked for (server-side running-request gauge).
6. Check every result against a number you can compute by hand.
7. Record GPU, driver, engine version, flags and commit with every result.
8. Separate what was measured from what is inferred, and state the limits.

## Budget log
| Date | Session | Spend |
|---|---|---|
| 2026-10-18 | Lab 05 baseline (H100, ~2.2 h) | $9.59 |
| | **Balance after** | **$30.41** of $40 |
| 2026-10-23 | Session A: Labs 01-04 (L40S, $1.10/hr; includes about 1 hour lost to a failed model cache and a pod migration) | _to fill in from the RunPod balance_ |

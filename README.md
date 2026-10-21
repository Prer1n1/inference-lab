# inference-lab

A hands-on, measured path through LLM inference performance, from GPU fundamentals to production serving, built to
cover the skills in Baseten's *Inference Performance* engineering posting. Every claim is backed by a benchmark in this repo,
every lab starts with written predictions, and mistakes found along the way are kept and explained.

The guiding question: **given one GPU and a latency target, what is the cheapest way to serve an LLM, and why does each setting change the numbers?**
Serving cost is `GPU $/hour ÷ tokens/hour`; every technique either raises tokens/hour or lowers latency at the same tokens/hour, and they trade against each other.

## What the job asks for, and where this repo answers it

| Posting item | Lab |
|---|---|
| Deep understanding of GPU architecture; reason from first principles about time and memory | 01, 04 |
| KV-cache management and reuse | 02, 08 |
| Continuous batching, LLM optimization basics | 03, 05 |
| Build benchmarking frameworks (models, batch sizes, sequence lengths, hardware) | `bench/`, 05, 15 |
| Profile end to end: launch overhead, memory layout, scheduling, tail-latency investigations | 06 |
| Quantization (FP8/FP4, AWQ, GPTQ) | 07 |
| Chunked prefill, custom scheduling | 08 |
| Speculative decoding | 09 |
| LoRA, guided generation for structured outputs | 10 |
| Cache-aware routing, prefill/decode disaggregation, load balancing | 11 |
| GPU kernels (CUDA, Triton, CUTLASS) | 12 |
| Bring up new architectures quickly | 13 |
| Distributed serving: autoscaling, observability, incident response | 14 |
| Tokens per GPU-hour, utilization, latency/throughput/cost tradeoffs | 05, 07, 15 |
| Contribute upstream to vLLM / SGLang / TensorRT-LLM | 16 |

## Lab status

| Lab | Topic | Status |
|---|---|---|
| [00](labs/00_concepts) | Concepts primer | Done |
| [01](labs/01_gpu_architecture) | GPU architecture and the roofline | Scripts ready |
| [02](labs/02_kv_cache) | KV cache | Scripts ready |
| [03](labs/03_batching) | Batching and context length | Scripts ready |
| [04](labs/04_prefill_decode) | Prefill vs decode against the roofline | Scripts ready |
| [05](labs/05_vllm_baseline) | vLLM baseline, concurrency sweep, cost per token | Measured; levels ≥128 need a re-run (client cap found) |
| [06](labs/06_profiling) | Profiling | Planned |
| [07](labs/07_quantization) | Quantization | Planned |
| [08](labs/08_scheduling_and_caching) | Chunked prefill, token budget, prefix caching | Planned |
| [09](labs/09_speculative_decoding) | Speculative decoding | Planned |
| [10](labs/10_structured_output_and_lora) | Structured output and multi-LoRA | Planned |
| [11](labs/11_routing_and_disaggregation) | Cache-aware routing, disaggregation | Planned |
| [12](labs/12_triton_kernel) | Triton kernel vs roofline | Planned |
| [13](labs/13_new_model_bringup) | New model bring-up | Planned |
| [14](labs/14_production_serving) | Kubernetes, autoscaling, observability | Planned |
| [15](labs/15_capacity_and_cost_report) | Capacity and cost report | Planned |
| [16](labs/16_upstream_contribution) | Upstream contribution | Planned |

**Highlights so far (Lab 05, Llama-3.1-8B on one H100, vLLM 0.32.0):** sharing the GPU among ~100 users made each output token about 44× cheaper
than serving one user ($6.91 → ~$0.16 per million output tokens); single-user decode reached 79% of the memory-bandwidth floor;
two benchmark bugs (prefix-cache contamination, a client-side connection cap) were found, fixed and guarded by tests.

## Layout
```
bench/            load generator (closed and open loop, goodput, GPU telemetry), mock server, plotting
tests/            harness self-tests that run against the mock server, no GPU needed
labs/NN_topic/    README (question, predictions, experiment, results, interpretation, takeaway, interview Qs), scripts, results/
templates/        the lab template every README follows
docs/             pod-workflow.md: how to run GPU sessions cheaply and safely
```

## Quick start (laptop, no GPU)
```
python -m venv .venv && .venv\Scripts\pip install -r requirements.txt      # Linux/macOS: .venv/bin/pip
.venv\Scripts\python tests\test_harness.py                                  # 5 tests against the mock server
.venv\Scripts\python bench\mock_server.py                                   # terminal 1
.venv\Scripts\python bench\bench.py --concurrency 1 8 32 --out scratch/run.jsonl   # terminal 2
```
Against a real server, point `--url` and `--model` at it. See `python bench/bench.py --help` for load models, SLO goodput, shared-prefix and output-length options.

## Running on a GPU
Follow [docs/pod-workflow.md](docs/pod-workflow.md). Labs 01-04 use `requirements-labs.txt`; vLLM labs use a separate environment (`requirements-vllm.txt`).
GitHub is the source of truth: edit locally, push, pull on the pod, run, copy results back, **terminate the pod**.

## Conventions (each one exists because it cost a measurement)
1. Write predictions, with the arithmetic, before running. Score them afterwards.
2. Fresh random prompts per level unless a lab is deliberately testing caching.
3. Warm up, then repeat. Two runs that disagree mean something is wrong; find out what before reporting either.
4. Report p50 and p99, and ranges when repeats disagree.
5. Verify that the load generator delivered the load you asked for (server-side running-request gauge).
6. Check every result against a number you can compute by hand.
7. Record GPU, driver, engine version, flags and commit with every result.

## Budget log
| Date | Session | Spend |
|---|---|---|
| 2026-10-18 | Lab 05 baseline (H100, ~2.2 h) | $9.59 |
| | **Balance after** | **$30.41** of $40 |

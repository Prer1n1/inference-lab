# inference-lab

## The problem

> Given one GPU and a latency target, what is the **cheapest way to serve an LLM**, and *why* does each setting change the numbers?

Serving cost is `GPU $/hour ÷ tokens produced per hour`. Every optimization (batching, quantization, KV-cache reuse,
speculative decoding) either raises tokens/hour or lowers latency at the same tokens/hour. But they trade against
each other: pushing throughput up makes individual users wait longer. Customers care about **both**, so the real
deliverable is a *curve* (throughput vs. latency) per configuration, and a recommendation for a given SLO.

Concretely, the end product is a report of the form:

> "For Llama-3.1-8B on 1×H100 with p99 ITL ≤ 50 ms, config X serves N tok/s at $Y per million tokens,
> 2.1× cheaper than the default. Here is the profile showing why."

That is exactly the work in Baseten's *Inference Performance* posting: profile, find what holds performance back,
turn it into tokens per GPU-hour, and report latency/throughput/cost tradeoffs.

## Roadmap

| Phase | What you do | What you learn | Needs GPU |
|---|---|---|---|
| 0 | Harness + mock server (done) | What TTFT/ITL/throughput mean and how to measure them honestly | no |
| 1 | Run vLLM baseline, sweep concurrency | The throughput/latency curve; where it saturates | yes |
| 2 | Profile one prefill and one decode step | Compute-bound vs memory-bound, in numbers | yes |
| 3 | Sweep configs: FP16 / FP8 / AWQ, prefix caching, chunked prefill | What each knob buys and costs | yes |
| 4 | Speculative decoding | Why decode is memory-bound and how drafts exploit it | yes |
| 5 | Triton kernel, benchmarked | What the engine does under the hood | yes |
| 6 | Cost report + upstream PR + write-up | Turning numbers into a decision | no |

Reading list and concepts: `docs/00-concepts.md`.

## Run the harness locally (no GPU)

```
python -m venv .venv && .venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python bench\mock_server.py          # terminal 1
.venv\Scripts\python bench\bench.py --concurrency 1 8 32   # terminal 2
```

Against a real server, point `--url` and `--model` at it; nothing else changes.

# Study plan: from here to interview-ready

Every lab follows the same loop: **read the mental model → write predictions → run → compare → explain → answer the interview questions aloud**.
A lab is *done* only when its README has real results, the predictions are scored, the limits and pitfalls are written down, and you can
answer its interview questions without notes.

## Where you are
- Done: Lab 00 (concepts), Lab 05 (measured; levels ≥128 need a re-run), laptop parts of Labs 09, 12 and 14, the benchmark tool, simulator and tests.
- Scripts ready, not yet run on a GPU: Labs 01-04, and the Lab 13 all-reduce benchmark.
- Lessons written, experiments not yet run: Labs 06-08, 10, 11, 15-21 and the GPU parts of 12-14.
- Budget: $30.41 of $40 left.

## Free work you can do now (laptop, no GPU)
| Do this | Lab | What you learn |
|---|---|---|
| Read Lab 00 and the *Mental model* of Labs 01-04, fill the **Predictions** tables | 00-04 | the theory before any measurement |
| Run `simulator.py validate / chunking / policies`; add a policy with **aging** (waiting time raises priority) and show it limits starvation | 09 | scheduler design, queueing, validating a model |
| Run `moe_math.py` on more MoE configs (for example DeepSeek-style); compare with the published sizes | 12 | total vs active parameters, bytes per step |
| Re-run `tokenizer_bench.py` on a quiet machine; try another tokenizer | 14 | CPU-side cost of long prompts |
| Write the router with three policies and fake replicas (Go preferred) and its tests | 16 | routing and load balancing, in Go |
| Learn Docker and a local Kubernetes (kind) with a dummy HTTP server | 19 | the base for the serving lab |
| Start the cost calculator using Lab 05 data | 20 | unit economics |
| Read the Further reading of each lab as you reach it | all | the papers behind the techniques |

## GPU sessions, in order
| Session | Labs | Where | Time | Rough cost | You finish with |
|---|---|---|---|---|---|
| A | 01-04 (`docs/session-a-labs-01-04.md`) | inexpensive 48 GB GPU | ~2 h | $1-3 | four labs measured; script bugs found |
| B | 05 re-run, 06 profiling | 1× H100 | ~3 h | ~$12 | corrected Lab 05; a profile of a step |
| C | 07 quantization, 08 scheduling and caching | 1× H100 (FP8 needs it) | ~4 h | ~$16 | two labs |
| D | 09 Part B, 10 speculative decoding, 11 guided decoding and LoRA | H100 or inexpensive | ~4 h | ~$10-16 | three labs |
| E | 12 Part B (MoE), 13 (multi-GPU) | 48 GB GPU; then 2 GPUs, ideally with and without NVLink | ~3-4 h | ~$15-30 | MoE and parallelism measured |
| F | 14 Part B (long context), 15 (engine comparison) | 1× H100 | ~5 h | ~$20 | long-context behavior; engine matrix |
| G | 16 (router on 2-3 replicas), 19 (Kubernetes, observability) | inexpensive GPUs | ~4-6 h | ~$10-20 | router results; a deployed, monitored service |
| H | 17 kernels, 18 bring-up | inexpensive GPU; H100 for 18 | ~5 h | ~$12-20 | kernels and a bring-up log |
| (no GPU) | 20 cost report, 21 upstream PR | laptop | | $0 | the capstone report; a pull request |

**Money:** sessions A-C fit in your current balance. Sessions D-H bring the whole plan to roughly **$100-140** in total, so plan a top-up of
about $80-100 over time, or use smaller models and cheaper GPUs wherever a lab allows (comparisons *within* a lab are valid on any GPU; only
absolute cost-per-token numbers need the H100 baseline). Order by value if money is short: 01-08, 10, 11, 15, 16, 18, 20, 21 first (they match
the posting's responsibilities); then 12-14, 17, 19.

## Rules that keep the work credible
1. Predictions are written before every run and scored after, even when wrong.
2. Check each result against a number you can compute by hand.
3. Two runs must agree before a number goes in a README; otherwise find out why.
4. Verify the load generator delivered the load you asked for.
5. Check that a comparison compares what you think it does (the "slow vs fast tokenizer" mistake in Lab 14 compared a thing with itself).
6. Keep the mistakes: they are the evidence that you debug measurements.
7. Terminate the pod when the session ends and update the budget log.

## Interview preparation, in parallel
- After each lab, answer its interview questions aloud in two minutes each. Record yourself once; it exposes vague explanations.
- Keep a running list titled "things that fooled me": cache contamination, the 100-connection client cap, the early-returning `asyncio.sleep`, the tokenizer that was not slow. Interviewers value these.
- After Lab 08 you can tell the full story: *measure → find the tail-latency problem → explain it with a trace → fix it → quantify the cost.*
- Before applying, publish two or three write-ups (for example: the roofline explanation of decode; the benchmarking pitfalls from Lab 05; the quantization trade-off table; the scheduler simulator) and keep the top-level README clean.
- When Labs 01-11 are done, do a mock interview: ask me to quiz you on any lab, one question at a time, without notes.

## Mapping to the roles you looked at
- **Inference Performance engineer**: Labs 01-18, 20, 21 are the core.
- **Infrastructure engineer**: Labs 16 and 19 (plus 05 for load testing); write the Lab 16 router in Go.
- **Observability engineer**: Lab 19 (metrics, tracing, SLOs) plus the harness's telemetry and goodput work; ClickHouse and telemetry pipelines are not covered.
- **Engineering manager (the first posting)**: needs prior team leadership. The labs build the technical credibility; leading or mentoring others (even in open source) builds the rest.

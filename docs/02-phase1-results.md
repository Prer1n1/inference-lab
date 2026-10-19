# Phase 1 results: vLLM baseline, Llama-3.1-8B-Instruct, 1x H100 SXM

## Environment
| | |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 (SXM), driver 580.126.09, CUDA 13.0 |
| Engine | vLLM 0.32.0 |
| Model | meta-llama/Llama-3.1-8B-Instruct, BF16 (15.0 GiB) |
| Flags | `--max-model-len 8192 --gpu-memory-utilization 0.90` (all else default) |
| KV cache | 53.47 GiB = 438,000 tokens (128 KiB/token) |
| Workload | ~256-word random-word prompts (~300 tokens) in, 128 tokens out, closed loop |
| Price | $3.99/hour (RunPod on-demand) |

## Clean results (fresh prompts, prefix cache hit rate 0%)
Throughput is the mean of repeats where available. Cost = $3.99 / (tokens per hour), output tokens only.

| Concurrency | Tokens/s | $ per M output tokens | ITL p50 | ITL p99 | TTFT p50 | TTFT p99 |
|---|---|---|---|---|---|---|
| 1 | 160 | 6.91 | 6.05 ms | 7.0 ms | 26 ms | 88 ms |
| 2 | 315 | 3.51 | 6.07 | 7.1 | 43 | 103 |
| 4 | 613 | 1.81 | 6.13 | 7.2 | 56 | 71 |
| 8 | 1,157 | 0.96 | 6.26 | 7.7 | 87 | 173 |
| 16 | 2,121 | 0.52 | 6.42 | 8.0 | 149 | 157 |
| 32 | 3,676 | 0.30 | 6.81 | 14.5 | 221 | 289-495 |
| 64 | 5,866 | 0.19 | 7.48 | 16.9-18.1 | 338-351 | 451-488 |
| 128 | 7,056 | 0.16 | 8.47 | 36-77 | 606-637 | 2,206-2,306 |
| 256 | 7,021 | 0.16 | 8.48 | 58-62 | 2,033-2,118 | 7,260-7,543 |

Repeats agree within ~1% on throughput (c=64, 128, 256). p99 ITL is noisy (c=128: 36-77 ms); medians are stable.

## Findings
1. Batching is the dominant lever: c=1 to c=128 gives 44x throughput and 44x lower cost per token.
2. Throughput plateaus at ~7,050 tok/s from c=128. c=256 adds nothing but doubles TTFT (2.1 s p50, 7.4 s p99).
3. Typical decode step (ITL p50) stays 6-8.5 ms at every level; the cost of load shows up as TTFT and tail ITL.
4. Under "p99 ITL <= 20 ms and p99 TTFT <= 0.5 s", the best operating point is c=64: ~5,900 tok/s at $0.19/M.
5. c=1 decode (6.05 ms) is 79% of the memory-bandwidth floor (16 GB / 3.35 TB/s = 4.8 ms).

## Methodology lesson: the first sweep was contaminated
`phase1_contaminated_cached_prompts.jsonl` used a fixed prompt seed, so identical prompts repeated across levels
and vLLM's prefix caching skipped their prefill. It overstated throughput by up to ~33% (9,369 vs ~7,050 tok/s at
c=128). Found by a second run that disagreed, fixed by generating fresh prompts per level (`bench.py`, commit 53fa0c5).
The server log confirmed 0.0% prefix cache hit rate during the clean runs. The reverse proof (identical prompts
give a high hit rate and higher throughput) is still to be demonstrated deliberately in Phase 3.

## Predictions vs reality
| | Predicted | Real |
|---|---|---|
| c=1 ITL | 6-8 ms | 6.05 ms |
| KV cache tokens | 350-450k | 438,000 |
| Throughput c=128 vs c=1 | 25-50x | 44x |
| Where it flattens | c = 100-300 | bends ~64, flat by 128 |

## Open questions (next phases)
- What is the saturating resource at c>=128: compute (prefill + decode), the scheduler's token budget, or a sequence cap? (Phase 2 profiling)
- What causes the p99 ITL spikes (36-77 ms vs 8.5 ms median)? Hypothesis: prefill work sharing a step with decode. (Phase 3: chunked prefill)

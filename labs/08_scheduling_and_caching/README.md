# Lab 08: scheduling and caching, chunked prefill, token budget, prefix reuse

**Status:** Planned (flags confirmed against the installed vLLM version when we run it)  
**Baseten posting items:** "chunked prefill", "KV-cache reuse", "custom scheduling and routing algorithms", tail latency  
**Needs:** 1× H100, about 2 hours (~$8) · **Depends on:** Labs 04, 05, 06

## 1. The question
Do chunked prefill, the per-step token budget and prefix caching tame the tail-latency spikes and lower cost, and by how much?

## 2. Concepts

### 2.1 The scheduler loop
Every step the scheduler:
1. Keeps all running sequences (one decode token each).
2. Admits waiting requests while limits allow: token budget (`max_num_batched_tokens`), sequence cap (`max_num_seqs`), free KV blocks.
3. If KV blocks run out, **preempts** a sequence (drop and recompute its cache later, or swap it out) and restarts it when space returns.
Policies decide *who* goes first: first-come-first-served is the default; priority and fairness policies are variations.

### 2.2 Chunked prefill
Without chunking, a long prompt is processed in one big step, during which no one else decodes: their next token is delayed by the whole prefill time (your p99 ITL spike). With chunked prefill, the prompt is split into chunks of at most the token budget, and each chunk shares a step with ongoing decode tokens ("piggybacking"). Effects:
- **ITL tail improves**: no step is dominated by one long prefill.
- **Long-prompt TTFT rises somewhat**: its prefill is spread over several steps.
- **Utilization improves**: decode tokens (memory-bound) ride along with prefill tokens (compute-bound), filling the GPU better.
The **token budget** is the dial: a large budget favors prefill speed, a small one favors steady ITL.

### 2.3 Prefix caching, in practice
Cached KV blocks are keyed by a hash of the tokens in the block and everything before it, so only an *identical prefix* up to a full-block boundary is reusable. Wins come from shared system prompts, few-shot examples, multi-turn conversations (the history is the next turn's prefix), and agent loops that resend large contexts. Blocks no running sequence uses stay cached until evicted (LRU). Consequences: **put stable content first and variable content last** in prompts; the cache hit rate depends on traffic, not on the engine alone, so a benchmark win may not appear in production unless real traffic shares prefixes.

### 2.4 Queueing basics
Latency stays flat until utilization nears capacity, then rises sharply (the "hockey stick"). Because requests vary in cost, tail latency worsens before the average does. Operating below the knee, with headroom, is a deliberate cost-vs-latency choice.

### 2.5 What Lab 05 left open
p99 ITL rose from 7 ms to 17 ms (64 users) to 36-77 ms (about 100 users) while the median barely moved. Hypothesis: prefill work sharing steps with decode. This lab tests it by changing the chunk/token budget and watching the tail.

## 3. Predictions (fill in BEFORE running)
| Quantity | Prediction | Real | Verdict |
|---|---|---|---|
| p99 ITL at 64 users, chunked prefill off vs on | | | |
| Effect of chunking on TTFT of long prompts | | | |
| Throughput change from a smaller token budget | | | |
| Cache hit rate and throughput gain with a 2,000-token shared prefix across all requests | | | |
| Concurrency ceiling set by `max_num_seqs` | | | |

## 4. Steps and why
1. **Build workloads** with the harness: (a) uniform short prompts (control), (b) mostly short plus some very long prompts (the interference case). *Chunking only matters when long and short mix.*
2. **Sweep chunked prefill on/off and the token budget** at a fixed open-loop rate; record p50/p99 TTFT and ITL, goodput under a latency target. *Open loop shows queueing; fixed rate makes configs comparable.*
3. **Re-trace a spike** (Lab 06 method) with chunking off and on. *Ties the metric change to the mechanism.*
4. **Prefix caching on purpose**: `--shared-prefix-words` at 0, 500, 2,000, 4,000; read the server's hit-rate gauge. *Turns the Lab 05 accident into a measurement.*
5. **Reproduce the contamination deliberately** (same seed twice) and explain the numbers.
6. **Probe `max_num_seqs`**: confirm via the harness's server-side gauge where the ceiling really is.

## 5. Results, 6. Interpretation, 7. Production takeaway, 8. Pitfalls
_To fill in._ Pitfalls: measuring the cache with prompts that do not share *full blocks*; comparing configs at different offered loads; forgetting that eviction can erase a cache you assumed warm.

## 9. Done when
- p99 ITL with and without chunked prefill, with trace evidence.
- A cache-hit-rate vs cost curve.
- A recommended scheduler configuration per workload shape.

## 10. Interview questions
1. Why does a long prompt hurt other users, and how does chunked prefill fix it? What does it cost?
2. Prefix caching gave a 30% win in a benchmark. How do you tell whether production traffic would see it?
3. What does `max_num_batched_tokens` trade off?
4. What happens when the KV cache fills, and which policy choices exist?

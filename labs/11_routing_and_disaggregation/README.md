# Lab 11: cache-aware routing and prefill/decode disaggregation

**Status:** Planned (flags and connector availability confirmed against the installed engine when we run it)  
**Baseten posting items:** "cache-aware routing", "custom scheduling and routing algorithms", "prefill/decode disaggregation", "load balancing"  
**Needs:** 2-3 inexpensive GPUs (or 2 H100 for the disaggregation test), about $6-10 · **Depends on:** Labs 05, 08

## 1. The question
Does routing by prompt-prefix affinity beat round-robin, and what does separating prefill from decode buy?

## 2. Concepts

### 2.1 Why load balancing LLM traffic is different
Classic web load balancing assumes requests cost about the same and replicas are interchangeable. Here:
- **Cost varies by orders of magnitude** (prompt and output lengths), so "equal request counts" is not equal load. Better signals: running requests, queued tokens, KV usage (replicas expose these as metrics).
- **Replicas are not interchangeable**: each holds a **prefix cache**. Sending a conversation's next turn to the replica that already has its prefix skips most of the prefill.
- **Cold starts are slow** (loading weights takes tens of seconds to minutes), so scaling reacts slowly.

### 2.2 Routing policies
| Policy | Idea | Weakness |
|---|---|---|
| Round-robin | rotate | ignores cost and cache locality |
| Least-loaded | pick the replica with the least outstanding work | ignores cache locality |
| Power of two choices | sample two replicas, pick the less loaded | cheap, good balance; no locality |
| Consistent hashing on a prefix | hash the prompt prefix to a replica | great locality; hot prefixes overload one replica |
| **Bounded-load affinity** | prefer the affinity replica unless it exceeds the average load by a margin, then spill to the next | locality with a balance guarantee |
The trade-off: **locality** (cache hits) vs **balance** (tail latency). The right margin is empirical.

### 2.3 Prefill/decode disaggregation
Prefill is compute-bound, decode memory-bound (Lab 04), and they interfere when colocated. **Disaggregation** runs them on separate GPU pools:
- **Prefill workers** process prompts, then send the resulting KV cache to
- **Decode workers**, which generate tokens.
Benefits: no prefill-induced ITL spikes; each pool is sized and tuned for its own bottleneck (different batch sizes, parallelism, even different GPU types). Costs: the **KV transfer** and engineering complexity.
Size of the transfer: `prompt_tokens × KV bytes per token`. For the 8B model, a 4,000-token prompt is about 512 MiB: at 50 GB/s (fast interconnect) about 10 ms; at 10 GB/s about 50 ms. That cost is paid once per request and must be small compared with the TTFT and ITL savings. Research systems that explored this include DistServe and Splitwise.
When it is **not** worth it: short prompts, low load, loose latency targets, or slow links.

### 2.4 Chunked prefill vs disaggregation
Chunked prefill (Lab 08) reduces interference *within* one GPU at no network cost; disaggregation removes it entirely at the price of a transfer and a second pool. They are alternatives for the same problem and are compared at equal GPU count.

## 3. Predictions (fill in BEFORE running)
| Quantity | Prediction | Real | Verdict |
|---|---|---|---|
| Cache hit rate: round-robin vs prefix affinity (multi-turn traffic, 3 replicas) | | | |
| TTFT p50 and p99 change from affinity | | | |
| KV transfer time for a 4,000-token prompt on your link | | | |
| Disaggregated vs 2 colocated replicas: ITL p99, throughput | | | |

## 4. Steps and why
1. **Write a small router** (Go or Python) in front of N replicas. Policies behind a flag: round-robin, least-loaded (poll replica metrics), bounded-load prefix affinity. *The routing logic is the deliverable and is testable without a GPU.*
2. **Unit tests with fake replicas**: policy decisions, spill-over behavior, handling a replica that dies. *Production routers fail in the edge cases.*
3. **Workload**: multi-turn conversations plus shared system prompts, open-loop arrivals. *Needs genuine prefix sharing to mean anything.*
4. **Measure per policy**: hit rate (replica gauge), TTFT p50/p99, tokens/s, load imbalance across replicas.
5. **Disaggregation prototype**: one prefill and one decode worker using the engine's KV-transfer support (confirm availability), vs two colocated replicas on the same hardware. *Equal GPU count is the only fair comparison.*
6. **Write up** when each wins and why.

## 5. Results, 6. Interpretation, 7. Production takeaway, 8. Pitfalls
_To fill in._ Pitfalls: benchmark traffic with no real prefix sharing; letting affinity pile hot prefixes on one replica; comparing at different GPU counts; ignoring router overhead and failure handling.

## 9. Done when
- Per-policy table: hit rate, TTFT p50/p99, tokens/s, imbalance.
- Disaggregated vs colocated at equal GPU count.
- The router code with tests.

## 10. Interview questions
1. Why can affinity routing hurt load balance, and how do you bound it?
2. What has to move between a prefill and a decode worker, and how big is it?
3. When is disaggregation not worth it?
4. What signals would you use to route by load instead of request count?

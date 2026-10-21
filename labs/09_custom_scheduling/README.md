# Lab 09: custom scheduling, inside the engine's scheduler

**Status:** Part A done (simulator, runs on a laptop). Part B (real engine) planned.  
**Baseten posting items:** "custom scheduling and routing algorithms", "working deep in runtime internals", "request scheduling" in the end-to-end profiling item  
**Needs:** Part A: a laptop. Part B: 1× inexpensive or H100 GPU, about 1-2 hours · **Depends on:** Labs 05, 08

## 1. The question
The scheduler decides, every few milliseconds, which requests run. What does the choice of policy change for latency and throughput, when does it matter, and what would it take to change it inside a real engine?

## 2. Mental model

### 2.1 The scheduler is a queueing system with an unusual server
- **Server**: the GPU running one engine step at a time. A step takes `base + per_seq × decoding_sequences + per_token × prefill_tokens` (the model fitted in 4.1).
- **Jobs**: requests with a prompt (prefill work) and an output (a number of decode steps) that is **unknown when the request arrives**.
- **Resources**: sequence slots (`max_num_seqs`), a per-step token budget, and KV-cache memory that each running request holds until it finishes.
- **Decisions each step**: who is admitted from the waiting queue, how much prefill each admitted request gets this step, whether to preempt someone when memory is short.

### 2.2 Why policy matters only when a queue exists
Ordering waiting requests changes nothing if nobody waits. Below capacity the queue is almost always empty, so every policy gives the same result. Near and above capacity a queue forms, and the policy decides *who* waits. This is the single most important idea in the lab, and Experiment 3 shows it.

### 2.3 The policies
| Policy | Rule | Strength | Weakness |
|---|---|---|---|
| **FCFS** | oldest first | simple, fair, no starvation | **head-of-line blocking**: one long prompt delays every short request behind it |
| **Shortest-prompt-first** | smallest prompt first | short requests are not stuck behind long ones; needs only information known on arrival | long requests can **starve** under sustained load |
| **Shortest-total-first (oracle)** | smallest prompt + output first | the best you could do with perfect knowledge | needs the output length, which nobody knows; a bound, not a real option |
| **Priority classes** | interactive before batch, FCFS inside a class | protects the latency-sensitive traffic you care about | batch work waits; needs a trustworthy way to classify |
Real systems add **aging** (waiting time raises priority, preventing starvation), per-tenant fairness, and deadlines.

### 2.4 Prefill and decode share the step (revisited from Lab 08)
- **Classic scheduling**: if any prompt is waiting, the step is prefill-only, and **decoding users stall** for its duration.
- **Chunked prefill**: prompts are split to fit a token budget and ride in the same step as decode tokens, so decoders keep moving.
The simulator implements both so the effect can be isolated.

### 2.5 What a simulator is for
A real engine takes minutes to restart and hours to benchmark. A simulator that reproduces the first-order behavior lets you try ten policies in seconds, **if it has been checked against reality**. It cannot discover effects it does not model (kernel behavior, cache effects, CPU overhead).

## 3. Prediction (before running)
1. Below capacity, all four policies give the same latency.
2. Just above capacity, FCFS chat latency explodes (long prompts block short ones) while shortest-prompt-first and priority keep it low.
3. Cost of those wins: the long batch jobs wait longer (their p99 grows).
4. Chunked prefill lowers the p99 inter-token time of decoding users when long and short prompts mix.
5. The calibrated simulator reproduces Lab 05's throughput within roughly 10% at user counts it was not fitted to.

## 4. Experiments (Part A: simulator, `simulator.py`)

### 4.1 Calibration and validation against Lab 05
```bash
python simulator.py validate
```
Constants: `base_ms = 6.05` (1 user), `per_seq_ms = 0.0244` (from 8.47 ms at about 100 sequences), `prefill_ms_per_token` fitted to the 64-user throughput. Only 1 and 64 users are used for fitting; the rest are held out.

| Users | Measured tokens/s (Lab 05) | Simulated | Error |
|---|---|---|---|
| 1 | 160 | 164 | +2.3% (fitted) |
| 8 | 1,157 | 1,213 | +4.9% |
| 16 | 2,121 | 2,161 | +1.9% |
| 32 | 3,676 | 3,853 | +4.8% |
| 64 | 5,866 | 6,047 | +3.1% (fitted) |
| ~100 | 7,056 | 7,606 | +7.8% |

Fitted prefill cost: 0.02 ms/token, i.e. about 50,000 prompt tokens/s, plausibly near the compute limit from Lab 04's formula (about 62,000 tokens/s at peak).
**Interpretation:** throughput is reproduced within 2-8% at held-out points, with a growing optimistic error at high load (the real engine pays overheads the model ignores). The single-user TTFT is not matched (simulated 12 ms vs measured 26 ms) because HTTP, tokenization and detokenization are not modelled. Good enough to rank policies; not good enough to quote absolute latencies.

### 4.2 Chunked prefill on a mixed workload
Workload: 80% chat (prompt about 300 tokens, output about 100) and 20% batch jobs (prompt about 3,000, output about 400), Poisson arrivals at 12 requests/s, FCFS, token budget 2,048.
```bash
python simulator.py chunking
```
| | Chat p99 time per output token | Overall p99 TTFT | Throughput |
|---|---|---|---|
| Classic (decoders stall during prefill) | 57.1 ms | 730 ms | 2,141 tok/s |
| Chunked prefill | **47.2 ms** | 770 ms | 2,141 tok/s |
**Interpretation:** chunking cut the chat users' tail inter-token time by about 17% at unchanged throughput, and made TTFT slightly worse (730 → 770 ms), the trade-off predicted in Lab 08. The size of the effect depends on how much long-prompt work shares the GPU.

### 4.3 Scheduling policies as load rises
```bash
python simulator.py policies --rates 16 20 30
```
| Arrival rate | Policy | Chat TTFT p50 / p99 (ms) | Batch E2E p99 (ms) | Tokens/s |
|---|---|---|---|---|
| 16/s | FCFS | 149 / 1,464 | 31,043 | 2,832 |
| 16/s | shortest-prompt | 128 / 1,433 | 31,043 | 2,832 |
| 16/s | priority | 133 / 1,433 | 31,043 | 2,832 |
| 20/s | FCFS | **3,723 / 7,615** | 71,641 | 3,345 |
| 20/s | shortest-prompt | 256 / 1,396 | 90,851 | 3,343 |
| 20/s | priority | 285 / 1,451 | 72,316 | 3,343 |
| 30/s | FCFS | **23,431 / 51,188** | 85,747 | 3,445 |
| 30/s | shortest-prompt | 251 / 1,688 | 139,336 | 3,331 |
| 30/s | priority | 346 / 1,224 | 94,209 | 3,404 |

**Interpretation:**
1. At 16/s (below capacity) the policies are indistinguishable: confirms 2.2.
2. At 20/s FCFS chat p50 is **3.7 s** against about **0.26 s** for shortest-prompt-first or priority: a 14× difference from reordering alone, at the *same* throughput.
3. The price: shortest-prompt-first makes the batch jobs' p99 **27% worse** at 20/s (72 → 91 s) and **63% worse** at 30/s (86 → 139 s); priority is much gentler on them because it only separates two classes. Both fix the chat tail; one pays for it in starvation.
4. Throughput is essentially unchanged by policy: scheduling redistributes *waiting*, it does not create capacity.
5. Overload at 30/s means FCFS has no good answer. The real fix is capacity or admission control (reject or shed load), not reordering.

### Limits of Part A
- A simulation, calibrated on one GPU, one model and one workload shape. Treat the ranking as a hypothesis for the real engine.
- Prompt and output lengths come from assumed lognormal distributions, not measured traffic.
- No preemption, no prefix cache, no speculative decoding, no CPU overhead.
- Prefill cost is linear in tokens; real attention cost grows with context.

## 5. Experiments (Part B: the real engine, planned)
1. **Read the scheduler** of the installed vLLM (and SGLang's, for contrast): where the waiting queue is ordered, where the token budget and slot limits are applied, where preemption happens. Write a one-page map with file and function names (confirm against the installed version).
2. **Use the built-in knobs first**: if the engine offers a priority scheduling policy, test it with the harness (open-loop, mixed workload) and compare with Part A's prediction.
3. **Make one small change**: for example, shortest-prompt-first ordering behind a flag, or aging for priority classes. Add a unit test for the ordering.
4. **Measure** FCFS vs your change at the same offered load. Does the real engine reproduce the simulated ranking? Where does it differ, and why?
5. **Record** what you learned about the engine's internals that the simulator got wrong.

## 6. Engineering takeaway
Scheduling policy is a **tail-latency and fairness tool, not a capacity tool**. It matters only when a queue forms, and then it decides which users bear the wait. The production decision is therefore about **objectives** (protect interactive traffic? bound the worst wait for everyone?) plus **admission control** for when demand exceeds capacity.

## 7. Interview questions
1. Why does the scheduling policy not matter below capacity, and what does that say about how to test it?
2. What is head-of-line blocking in LLM serving, and how do shortest-first and priority scheduling address it?
3. Why can't a scheduler use shortest-total-job-first in practice, and what can it use instead?
4. How would you prevent starvation of long requests under a shortest-first policy?
5. What did you do to check the simulator before trusting it, and what does it still not capture?

## Further reading
- Orca: "A Distributed Serving System for Transformer-Based Generative Models" (Yu et al., OSDI 2022), iteration-level scheduling.
- Sarathi-Serve (Agrawal et al., 2024): chunked prefill and stall-free scheduling.
- vLLM paper (Kwon et al., SOSP 2023): block-based memory management and scheduling.
- Any queueing-theory primer on shortest-job-first, processor sharing and the hockey-stick latency curve.

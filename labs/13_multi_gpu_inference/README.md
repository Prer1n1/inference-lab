# Lab 13: multi-GPU inference, tensor, pipeline and expert parallelism

**Status:** Scripts ready (all-reduce benchmark); arithmetic done; GPU measurements planned  
**Baseten posting items:** "large-scale distributed serving", "bring up new model architectures on new hardware", Baseten's own example projects on multi-node inference and B200 GPUs; the "deep understanding of GPU architecture" requirement at system level  
**Needs:** 2 GPUs on one machine, about 1.5-2 hours (about twice the single-GPU hourly rate; run once with and once without NVLink if budget allows) · **Depends on:** Labs 01, 05, 12

## 1. The question
When one GPU is not enough, how do you split a model across several, what does the splitting cost in communication, and when are two replicas better than one model spread over two GPUs?

## 2. Mental model

### 2.1 Why split a model at all
1. **It does not fit.** Llama-3.1-70B in BF16 needs about **131.5 GiB** for weights alone; an 80 GB GPU holds 80. FP8 halves that to 65.8 GiB, which fits one H100 only with almost no room for KV cache. Mixtral-8x7B (87 GiB, Lab 12) is in the same position.
2. **Latency.** Decode time is bounded by `weight bytes read per GPU ÷ bandwidth`; spreading weights over N GPUs divides the bytes each GPU reads.
3. **KV capacity.** Each GPU holds only its share of weights and KV, leaving more memory per GPU for cache.

### 2.2 The parallelism families
| Kind | What is split | Communication | Best for |
|---|---|---|---|
| **Tensor parallelism (TP)** | every weight matrix, across GPUs (Megatron style) | **two all-reduces per transformer layer** on the activations | latency, fitting a model inside one fast-linked node |
| **Pipeline parallelism (PP)** | groups of consecutive layers go to different GPUs | activations passed between stages once per stage boundary | across nodes with slower links; adds latency and needs micro-batching to avoid idle "bubbles" |
| **Expert parallelism (EP)** | MoE experts distributed over GPUs (Lab 12) | **all-to-all** twice per MoE layer (send tokens to experts, bring results back) | large MoE models |
| **Data parallelism / replicas** | nothing split; N identical copies | none between replicas (a router spreads requests, Lab 16) | throughput when the model fits |
| **Context / sequence parallelism** | one long sequence's tokens (Lab 14) | key/value blocks exchanged between GPUs | very long prompts |
Real deployments combine them, for example TP inside a node and PP or EP across nodes.

### 2.3 How tensor parallelism works
For an MLP `y = W2 · act(W1 · x)`: split `W1` by **columns** across GPUs (each GPU computes its slice of the hidden activations without talking), then split `W2` by **rows** (each GPU produces a partial sum of the output); one **all-reduce** adds the partial sums. Attention does the same: heads are divided across GPUs, and the output projection ends with an all-reduce. So each layer needs **two all-reduces** in the forward pass. KV heads are split too: with 8 KV heads, TP up to 8 divides them evenly; beyond that KV must be replicated.

### 2.4 The cost of an all-reduce
A **ring all-reduce** on N GPUs moves `2(N−1)/N × message_bytes` per GPU and takes about `latency + that ÷ link bandwidth`.
- During decode, messages are **small** (`batch × hidden × 2 bytes`). For a 70B-class model (hidden 8,192): batch 1 is 16 KiB, batch 64 is 1 MiB.
- Small messages are **latency-bound**: dominated by a fixed cost per call (tens of microseconds is typical on a fast link; this is exactly what `allreduce_bench.py` measures), not by bandwidth.
- A step needs `2 × layers` all-reduces: 160 for an 80-layer model. If each takes about 15 µs (an assumption to replace with your measurement), communication adds about **2.4 ms per step**, against a weight-read floor of about 5.3 ms per GPU at TP=8 (17.6 GB ÷ 3.35 TB/s). That is a **45% overhead**, which is why TP speedups for decode are well below N×.
- Rule of thumb: communication matters more as GPUs get faster at math and as batches get smaller.

### 2.5 Interconnects (nominal figures; check your hardware)
| Link | Rough bandwidth per direction |
|---|---|
| NVLink 4 (H100 SXM) | about 450 GB/s (900 GB/s bidirectional total) |
| PCIe Gen5 x16 | about 64 GB/s |
| PCIe Gen4 x16 | about 32 GB/s |
| InfiniBand NDR 400 Gb/s port | about 50 GB/s |
That roughly 7× gap between NVLink and PCIe is why TP is normally kept **inside** an NVLink-connected node, and why cheaper PCIe-only multi-GPU machines scale TP poorly.

### 2.6 Replicas vs tensor parallelism at the same GPU count
If the model fits on one GPU, two options for two GPUs:
- **Two TP=1 replicas** behind a router: nearly 2× the throughput, no communication, but each user's decode speed is that of one GPU.
- **One TP=2 replica**: roughly half the weight bytes per GPU (lower latency per user, minus communication) and more KV capacity, but less than 2× throughput because of all-reduce overhead.
Rule: **use the least parallelism that satisfies fit and latency, then scale out with replicas.**

### 2.7 NCCL
NVIDIA's collective-communication library (all-reduce, all-gather, all-to-all, send/recv) that frameworks use. Message size, topology (NVLink, PCIe switches, network) and algorithm (ring, tree) determine speed. `nccl-tests` and this lab's script measure it directly. Hangs and slowdowns usually trace to topology or environment settings, so know how to check them (`nvidia-smi topo -m` shows the link types between GPUs).

## 3. Predictions (fill in BEFORE running)
| Quantity | Prediction | Real | Verdict |
|---|---|---|---|
| 2-GPU all-reduce `busbw` for a 1 GiB message: NVLink vs PCIe machine | | | |
| All-reduce time for a 512 KiB message (decode-sized) | | | |
| Latency-bound region: size below which time barely changes | | | |
| TP=2 vs TP=1 ITL at 1 user (8B model; floor halves from about 4.8 ms) | | | |
| TP=2 throughput at 64 users vs 2 × TP=1 replicas | | | |
| 70B BF16 weights per GPU at TP=2, 4, 8 (from 131.5 GiB) | 65.8 / 32.9 / 16.4 GiB | | |

## 4. Experiments

### 4.1 Arithmetic (done)
Checked: 70B BF16 = 131.5 GiB (FP8 65.8 GiB); 70B KV = 320 KiB/token; weights per GPU at TP=1/2/4/8 = 131.5 / 65.8 / 32.9 / 16.4 GiB; decode all-reduce messages for hidden 8,192 are 16 KiB, 256 KiB and 1 MiB at batch 1, 16, 64.

### 4.2 Measure the interconnect (planned)
```bash
nvidia-smi topo -m                                           # how the GPUs are connected
torchrun --nproc_per_node=2 allreduce_bench.py --out results/allreduce_2gpu.json
```
It sweeps message sizes from 8 KiB to 1 GiB and prints time, algorithmic bandwidth and bus bandwidth. *Why:* the latency at decode-sized messages is the number that decides whether TP is worth it, and it lets you replace the 15 µs assumption in 2.4 with a measurement. If you can rent both an NVLink machine and a PCIe machine, run both: the contrast is the lesson.

### 4.3 TP=1 vs TP=2 for the 8B model (planned)
Serve with `--tensor-parallel-size 1` and `2` (flag name to be confirmed for the installed version), sweep 1, 16, 64 users with the harness, and record ITL, TTFT and tokens/s. Compare ITL at 1 user with the floors (4.8 ms vs about 2.4 ms plus communication).

### 4.4 Replicas vs TP at equal GPUs (planned)
Compare one TP=2 server with two TP=1 servers (two ports) driven by the harness at the same total offered load, using the matrix runner. Report throughput, ITL p50/p99 and goodput at a latency target. Decide, from numbers, which wins at which load.

### 4.5 Optional, budget permitting
70B in FP8 on TP=2, 4, 8 (more GPUs, more money); expert parallelism for an MoE if the engine supports it; a pipeline-parallel run across two machines.

## 5. Engineering takeaway
Parallelism is a **fit-and-latency tool with a communication tax**, not a free speedup. The tax depends on link speed and message size; decode's small messages make it a latency problem rather than a bandwidth problem. Choose the **smallest tensor-parallel degree** that fits the model and meets the latency target, keep TP inside the fastest links, and scale throughput with replicas.

## 6. Interview questions
1. Why does tensor parallelism need two all-reduces per layer?
2. Why is a decode-time all-reduce latency-bound rather than bandwidth-bound?
3. When would you choose two TP=1 replicas over one TP=2 replica?
4. Why is TP kept inside a node and pipeline parallelism used across nodes?
5. What goes wrong when TP size exceeds the number of KV heads?
6. What does `nvidia-smi topo -m` tell you, and why does it matter?

## Further reading
- Megatron-LM (Shoeybi et al., 2019): tensor parallelism for transformers.
- GPipe (Huang et al., 2018) and PipeDream: pipeline parallelism and bubbles.
- NCCL documentation and the `nccl-tests` repository.
- DeepSeek-V3 technical report for expert-parallel serving at scale.

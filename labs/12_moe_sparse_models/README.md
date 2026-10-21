# Lab 12: mixture of experts (sparse models)

**Status:** Part A done (arithmetic from configs, runs on a laptop). Part B (GPU measurements) planned.  
**Baseten posting items:** "Bring up and tune new model architectures" (many new frontier models are MoE); "expert" and multi-node serving context in Baseten's own example projects  
**Needs:** Part A: a laptop. Part B: a 48 GB GPU, about 1.5 hours (~$2-5) · **Depends on:** Labs 01, 03, 05

## 1. The question
A mixture-of-experts model has far more parameters than it uses per token. How does that change memory, decode speed and throughput compared with a dense model, and when does the advantage disappear?

## 2. Mental model

### 2.1 What an MoE layer is
In a dense transformer every token passes through one feed-forward network (MLP) per layer. An **MoE layer** replaces it with `E` parallel MLPs called **experts** plus a small **router**:
1. The router (a linear layer) scores every expert for the token.
2. The token is sent to its **top-k** experts (for example 2 of 8, or 4 of 60 plus always-on **shared** experts, or 8 of 256).
3. The experts' outputs are combined, weighted by the router scores.
Attention layers stay dense. Experts are sized so each is a normal MLP, which is why total parameters multiply while per-token compute does not.

### 2.2 Total vs active parameters
- **Total parameters** decide **memory**: every expert must be resident, because any token may need any of them.
- **Active parameters** (what one token uses) decide **compute per token**.
Examples reproduced from configs by `moe_math.py` (and matching the published figures):

| Model | Experts / top-k | Total | Active per token | Active share | BF16 weights |
|---|---|---|---|---|---|
| Qwen1.5-MoE-A2.7B | 60 routed + 1 shared (4× an expert) / top-4 | 14.32B | 2.69B | 19% | 26.7 GiB |
| Mixtral-8x7B | 8 / top-2 | 46.70B | 12.88B | 28% | 87.0 GiB |
A 14B-parameter MoE does the arithmetic of a roughly 2.7B model per token but needs the memory of a 14B model. Mixtral does not fit one 80 GB GPU in BF16.

### 2.3 Decode is memory-bound, so what matters is which weights are *read*
Each decode step reads the weights of every expert that **at least one token in the batch** selected. With independent uniform routing, the expected number of distinct experts touched is
`touched = E × (1 − (1 − k/E)^B)` for batch size `B`.

| Batch | Qwen MoE (E=60, k=4) touched | Bytes read per step | Floor on H100 (3.35 TB/s) | Mixtral (E=8, k=2) touched | Floor |
|---|---|---|---|---|---|
| 1 | 4.0 | 5.4 GB | 1.61 ms | 2.0 | 7.69 ms |
| 8 | 25.5 | 14.3 GB | 4.26 ms | 7.2 | 25.19 ms |
| 32 | 53.4 | 25.9 GB | 7.73 ms | 8.0 | 27.88 ms |
| 64 | 59.3 | 28.3 GB | 8.46 ms | 8.0 | 27.88 ms |
| dense model of the same total size | — | 28.6 GB | **8.55 ms** | — | **27.88 ms** |

**The key insight:** at batch 1 the MoE reads only the active experts, so decode is about **5× faster** than a dense model of the same total size (Qwen: 1.61 vs 8.55 ms). As batch grows, almost every expert gets touched each step, so the **bytes read converge to the full model's**, and the memory-bound advantage **vanishes by batch 32-64**. A small expert count (Mixtral's 8) loses it by batch 8.

### 2.4 Where the MoE advantage survives at high batch
Once decode is compute-bound (Lab 03), per-token FLOPs matter and the MoE still does about a fifth of the math of a same-size dense model. So MoE wins on **latency at low batch** (fewer bytes) and **throughput at high batch** (fewer FLOPs per token); it costs **memory capacity** and adds **routing, load-balance and communication** problems.

### 2.5 Load balance
Routers can send most tokens to a few popular experts. In training an auxiliary loss pushes toward balance; at inference, imbalance means (a) the touched-experts count is lower than the uniform formula (fewer bytes, a pleasant surprise) and (b) under **expert parallelism** some GPUs get more work than others and everyone waits for the slowest.

### 2.6 Serving MoE at scale
- **Expert parallelism (EP)**: experts are partitioned across GPUs. Each layer does an **all-to-all** to send tokens to the GPUs that hold their experts and another to bring results back; communication scales with tokens × hidden size (Lab 13).
- **Fused MoE kernels**: running many small expert matmuls efficiently needs grouped/fused GEMM kernels that batch tokens per expert (Triton or CUTLASS implementations in the engines).
- **Quantization** matters more because total size is large (FP8 halves Mixtral's footprint to fit two GPUs more easily).

## 3. Predictions (before the GPU measurements in Part B)
| Quantity | Prediction (arithmetic above) | Real | Verdict |
|---|---|---|---|
| Qwen MoE ITL at 1 user vs a dense model with the same *total* size | about 5× lower | | |
| Qwen MoE ITL at 1 user vs a dense model with the same *active* size | similar or slightly higher (more overhead: router, scattered expert reads) | | |
| ITL growth for the MoE from 1 to 64 users | grows toward the same-total-size dense model | | |
| Throughput at 64+ users vs the dense model with the same total size | higher (fewer FLOPs per token) | | |
| Whether the touched-experts count is below the uniform estimate (router skew) | below | | |

## 4. Experiments

### 4.1 Part A: the arithmetic (done)
```bash
python moe_math.py Qwen/Qwen1.5-MoE-A2.7B mistralai/Mixtral-8x7B-v0.1 --bandwidth-gb-s 3350
```
Full output: `results/moe_math_output.txt`. **Check:** the script computes parameters from config fields alone and reproduces the published totals (Qwen 14.3B total / 2.7B active, Mixtral 46.7B / 12.9B), which is the reason to trust its byte counts.
**Limits:** assumes uniform independent routing and ignores attention-cache bytes, activations and kernel overheads. Floors are lower bounds, and ceilings in the script's table ignore compute, so they are meaningless once decode turns compute-bound (the batch-128 "ceiling" of 14,979 tokens/s is not a prediction).

### 4.2 Part B: GPU measurements (planned)
1. Serve Qwen1.5-MoE-A2.7B (28.6 GB in BF16; fits a 48 GB card) with the Lab 05 settings and harness.
2. Serve two dense comparators in the same engine: a ~3B dense model (same *active* size) and a ~14B dense model (same *total* size). Use the closest available family and note the differences (architecture, tokenizer, training), since no pair is perfectly matched.
3. Sweep 1, 4, 16, 64 users with fresh prompts and the upgraded harness (server-side load check); record ITL p50/p99 and tokens/s.
4. Compare with Section 2.3's floors.
5. **Router skew**: run a script with the Hugging Face reference model on realistic text and count, per layer, how often each expert is chosen (a hook on the router). Plot the histogram and compare the touched-experts count with the uniform formula.
6. If the engine exposes MoE kernel choice or a profile (Lab 06), find the fused-MoE kernel's share of the step.

## 5. Engineering takeaway
For serving, an MoE model's **cost structure differs from a dense model's**: capacity is priced by *total* parameters, per-token compute by *active* parameters, and decode bandwidth by something between the two that depends on batch size. The same GPU can show a 5× decode advantage at batch 1 and almost none at batch 64. Always state the batch size when quoting an MoE speedup, and size GPUs by total memory first.

## 6. Interview questions
1. Why does an MoE model need memory for all its parameters but compute for only a few?
2. Derive the expected number of experts touched per step at batch size B, and say when the memory-bound advantage over a dense model disappears.
3. What is expert parallelism and what communication does it need?
4. How does router imbalance hurt an expert-parallel deployment?
5. Why do MoE models benefit especially from quantization?

## Further reading
- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (2017).
- Mixtral of Experts (Jiang et al., 2024).
- DeepSeek-V3 technical report: fine-grained experts, shared experts and expert-parallel serving.
- Switch Transformers (Fedus et al., 2021) for load-balancing losses.

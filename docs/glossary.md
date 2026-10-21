# Glossary

Short definitions, with the lab where each idea is worked out. Alphabetical.

| Term | Meaning | Lab |
|---|---|---|
| **All-reduce** | A collective operation that sums a tensor across GPUs and gives every GPU the result; tensor parallelism does two per layer | 13 |
| **All-to-all** | Every GPU sends a different chunk to every other GPU; expert parallelism uses it to route tokens | 12, 13 |
| **Arithmetic intensity** | FLOPs performed per byte moved; decides memory- vs compute-bound | 01 |
| **AWQ / GPTQ** | Post-training methods that quantize weights to about 4 bits while protecting accuracy | 07 |
| **Batch size (decode)** | Number of sequences that share one engine step | 03 |
| **Chunked prefill** | Splitting a long prompt into pieces that share steps with decoding users | 08 |
| **Closed loop / open loop** | Fixed number of users in flight / requests arriving at a rate regardless of completions | 05 |
| **Continuous batching** | Re-forming the batch every step so finished requests leave and new ones join | 03, 05 |
| **Context length** | Number of tokens (prompt plus generated) a sequence holds | 02, 14 |
| **CUDA graph** | A recorded sequence of kernel launches replayed with one launch to cut CPU overhead | 06 |
| **Decode** | Generating output tokens one at a time; memory-bound | 04 |
| **Disaggregation (prefill/decode)** | Running prefill and decode on separate GPU pools and transferring the KV cache between them | 16 |
| **E2E latency** | Request sent to last token received | 05 |
| **Expert (MoE)** | One of several parallel feed-forward networks in a mixture-of-experts layer | 12 |
| **Expert parallelism** | Distributing MoE experts across GPUs | 12, 13 |
| **FLOP / FLOP/s** | A floating-point operation / operations per second | 01 |
| **FP8 / FP4 / INT4** | 8-bit float, 4-bit float, 4-bit integer number formats | 07 |
| **GEMM / GEMV** | Matrix-matrix / matrix-vector multiply; prefill is GEMM-shaped, batch-1 decode is GEMV-shaped | 01, 17 |
| **Goodput** | Throughput counting only requests that met the latency targets | 05 |
| **GQA** | Grouped-query attention: several query heads share one key/value head, shrinking the KV cache | 02 |
| **HBM** | High-bandwidth memory: the GPU's main memory | 01 |
| **ITL / TPOT** | Gap between streamed tokens / mean time per output token after the first | 05 |
| **KV cache** | Stored keys and values of earlier tokens so they are not recomputed | 02 |
| **Kernel** | A function that runs on the GPU | 01, 06 |
| **Little's law** | Average requests in system = arrival rate × average time in system | 05, 20 |
| **LoRA** | Low-rank adapters that modify a frozen base model with a small number of extra parameters | 11 |
| **MBU / MFU** | Memory-bandwidth utilization / model-FLOPs utilization: achieved ÷ peak | 04, 06 |
| **MoE** | Mixture of experts: a layer with many experts of which each token uses a few | 12 |
| **NCCL** | NVIDIA's multi-GPU collective-communication library | 13 |
| **NVLink** | Fast direct GPU-to-GPU link, several times faster than PCIe | 13 |
| **Occupancy** | How many warps are resident on an SM relative to its maximum | 17 |
| **PagedAttention** | Storing the KV cache in fixed-size blocks mapped through a table, like virtual memory | 02 |
| **p50 / p99** | The median / the value 99% of requests beat; p99 is the tail | 05 |
| **Pipeline parallelism** | Splitting a model's layers across GPUs in stages | 13 |
| **Prefill** | Processing all prompt tokens in one pass; compute-bound; sets TTFT | 04 |
| **Prefix caching** | Reusing the KV blocks of an identical prompt prefix | 02, 08 |
| **Quantization** | Storing numbers in fewer bits to cut memory and bandwidth | 07 |
| **RadixAttention** | SGLang's prefix cache organized as a radix tree | 15 |
| **Ridge point** | Peak FLOP/s ÷ peak bytes/s: the arithmetic intensity where a kernel stops being memory-bound | 01 |
| **Roofline** | Model: attainable FLOP/s = min(peak, intensity × bandwidth) | 01 |
| **Scheduler** | The engine component choosing which requests run each step | 05, 08, 09 |
| **SLO / SLI** | Service-level objective (target) / indicator (the measured quantity) | 19 |
| **SM** | Streaming multiprocessor: the GPU's basic execution unit | 01 |
| **Speculative decoding** | A cheap draft guesses tokens; the target verifies several at once | 10 |
| **Tensor core** | Hardware unit performing small matrix multiply-accumulates, the source of LLM throughput | 01 |
| **Tensor parallelism** | Splitting every weight matrix across GPUs | 13 |
| **Token budget** | The maximum number of tokens processed in one engine step | 08 |
| **TTFT** | Time to first token: queueing plus prefill | 05 |
| **Warp** | 32 GPU threads executing in lockstep | 01, 17 |

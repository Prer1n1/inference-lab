# Lab 12: A Triton kernel against the roofline

**Status:** Planned  
**Baseten posting items:** GPU kernels (CUDA, Triton, CUTLASS); deep GPU architecture understanding  
**Needs:** inexpensive GPU, ~1-2 h (~$3) · **Depends on:** Labs 01, 04

## 1. The question
Can I write a kernel that matches or beats PyTorch on an inference operation and explain the result with the roofline?

## 2. Plan
1. Fused RMSNorm with residual add in Triton; correctness tests against PyTorch (`allclose`) over shapes and dtypes.
2. Second kernel: a weight-only INT8/FP8 dequantize-GEMV (the decode case) or a fused softmax.
3. Benchmark against PyTorch eager and `torch.compile` across sizes; report achieved GB/s as a % of the Lab 01 measured bandwidth.
4. Optional: compare with the corresponding vLLM kernel.

## 3. Predictions (write before running)
| Quantity | Prediction (with arithmetic) | Real | Verdict |
|---|---|---|---|


## 4. Done when
- Kernel code, tests, and a bandwidth-vs-size plot
- A paragraph explaining where the kernel is memory-bound and what limits it

## 5. Results, interpretation, production takeaway, pitfalls
_To fill in using `templates/LAB_TEMPLATE.md`: raw data in `results/`, p50 and p99, repeats, and a check against a hand calculation._

## 6. Interview questions
1. Why is fusing elementwise ops with a norm a win?
2. How do you know whether a kernel is memory- or compute-bound?
3. What does tile size trade off in a matmul kernel?

"""Lab 01: what is this GPU, and what can it actually do?

Measures (not quotes) the two numbers every later lab is compared against:
  - memory bandwidth  -> sets the floor on decode step time (weights must be read every step)
  - matmul throughput -> sets the ceiling on prefill speed
and their ratio, the roofline "ridge point" (FLOPs per byte where a kernel stops being memory-bound).

    python gpu_info.py --params-b 8 --out results/gpu_info.json
Needs a CUDA GPU and PyTorch.
"""
import argparse
import json
import os

import torch


def seconds_per_call(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1e3


def measure_bandwidth_gbs(gib=2):
    n = gib * 2**30 // 2  # bf16 elements
    x = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    y = torch.empty_like(x)
    t = seconds_per_call(lambda: y.copy_(x))
    return 2 * x.numel() * 2 / t / 1e9  # one read + one write


def measure_tflops(dtype, n=8192):
    a = torch.randn(n, n, dtype=dtype, device="cuda")
    b = torch.randn(n, n, dtype=dtype, device="cuda")
    t = seconds_per_call(lambda: a @ b)
    return 2 * n**3 / t / 1e12


def memory_accounting_demo():
    """Why nvidia-smi and PyTorch disagree: CUDA context, allocator caching.

    The earlier benchmarks leave blocks in PyTorch's cache; without emptying it first, the 1 GiB tensor is
    served from that cache and free memory never moves (seen in the first L40S run). Empty it first.
    """
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free0, total = torch.cuda.mem_get_info()
    x = torch.empty(2**30, dtype=torch.uint8, device="cuda")  # 1 GiB
    free1, _ = torch.cuda.mem_get_info()
    del x
    free2, _ = torch.cuda.mem_get_info()  # freed to PyTorch's cache, not to the driver
    reserved_after_del = torch.cuda.memory_reserved()
    torch.cuda.empty_cache()
    free3, _ = torch.cuda.mem_get_info()  # now returned to the driver
    gib = lambda b: round(b / 2**30, 3)  # noqa: E731
    return {"total_gib": gib(total), "free_before_gib": gib(free0), "free_with_1gib_tensor_gib": gib(free1),
            "free_after_del_gib": gib(free2), "free_after_empty_cache_gib": gib(free3),
            "allocated_after_del_gib": gib(torch.cuda.memory_allocated()),
            "reserved_after_del_gib": gib(reserved_after_del)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params-b", type=float, default=8.0, help="model size in billions of parameters")
    ap.add_argument("--out", default="results/gpu_info.json")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "no CUDA GPU visible"

    p = torch.cuda.get_device_properties(0)
    bw = measure_bandwidth_gbs()
    tf = {str(dt).split(".")[-1]: round(measure_tflops(dt), 1) for dt in (torch.bfloat16, torch.float16)}
    best_tf = max(tf.values())
    info = {
        "name": p.name, "compute_capability": f"{p.major}.{p.minor}", "sm_count": p.multi_processor_count,
        "total_memory_gib": round(p.total_memory / 2**30, 2),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "measured_bandwidth_gb_s": round(bw, 1),
        "measured_matmul_tflops": tf,
        "ridge_point_flop_per_byte": round(best_tf * 1e12 / (bw * 1e9), 1),
        "memory_accounting_gib": memory_accounting_demo(),
    }
    for bytes_per_param, name in ((2, "bf16"), (1, "fp8/int8"), (0.5, "int4")):
        weight_gb = a.params_b * bytes_per_param
        info.setdefault("decode_step_floor_ms", {})[name] = round(weight_gb / bw * 1e3, 2)
        info.setdefault("weights_gb", {})[name] = weight_gb
    print(json.dumps(info, indent=2))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()

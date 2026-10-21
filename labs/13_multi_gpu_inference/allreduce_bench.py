"""Lab 13: measure collective-communication speed between GPUs (all-reduce), the cost tensor parallelism pays per layer.

    torchrun --nproc_per_node=2 allreduce_bench.py --out results/allreduce_2gpu.json
Needs 2+ CUDA GPUs, PyTorch with NCCL. Rank 0 prints and saves a table.

For each message size it reports
  time   per all-reduce (microseconds)
  algbw  message bytes / time
  busbw  algbw * 2*(n-1)/n   -- the per-link bandwidth a ring all-reduce actually used; comparable to the hardware's
         link speed and to nccl-tests output
Small messages are latency-bound (time barely changes with size); large ones are bandwidth-bound. Decode-time
tensor parallelism sends SMALL messages (batch x hidden x 2 bytes, for example 64 x 4096 x 2 = 512 KiB), twice per layer.
"""
import argparse
import json
import os

import torch
import torch.distributed as dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-log2", type=int, default=13, help="smallest message = 2**N bytes (13 = 8 KiB)")
    ap.add_argument("--max-log2", type=int, default=30, help="largest message = 2**N bytes (30 = 1 GiB)")
    ap.add_argument("--out", default="results/allreduce.json")
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    rows = []
    for log2 in range(a.min_log2, a.max_log2 + 1):
        nbytes = 2**log2
        x = torch.ones(nbytes // 2, dtype=torch.bfloat16, device="cuda")
        iters = 200 if nbytes <= 2**22 else 30
        for _ in range(10):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            dist.all_reduce(x)
        end.record()
        torch.cuda.synchronize()
        t = start.elapsed_time(end) / iters / 1e3
        algbw = nbytes / t / 1e9
        rows.append({"bytes": nbytes, "time_us": round(t * 1e6, 1), "algbw_gb_s": round(algbw, 2),
                     "busbw_gb_s": round(algbw * 2 * (world - 1) / world, 2)})
        del x
    if rank == 0:
        print(f"world size {world}, GPU: {torch.cuda.get_device_name(0)}")
        print(f"{'bytes':>12} {'time us':>9} {'algbw GB/s':>11} {'busbw GB/s':>11}")
        for r in rows:
            print(f"{r['bytes']:>12} {r['time_us']:>9} {r['algbw_gb_s']:>11} {r['busbw_gb_s']:>11}")
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"world": world, "gpu": torch.cuda.get_device_name(0), "rows": rows}, f, indent=2)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

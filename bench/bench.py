"""Load generator + metrics for any OpenAI-compatible streaming endpoint (vLLM, SGLang, mock).

Load models
  closed-loop  --concurrency N [N ...]  keep exactly N requests in flight
  open-loop    --rate R [R ...]         Poisson arrivals at R requests/s, regardless of completions;
                                        exposes queueing collapse that closed-loop hides
Metrics (latencies in ms), per request then aggregated as p50/p99
  TTFT  time to first token        queueing + prefill
  TPOT  (E2E - TTFT) / (output_tokens - 1)   mean decode speed one user sees
  ITL   gap between streamed chunks          equals the token gap only if each chunk is one token
                                             (not true with speculative decoding)
  E2E   whole request
  out_tok_per_s   output tokens (server-reported usage) / wall time
  goodput         share of requests meeting --slo-ttft-ms and --slo-tpot-ms, and the tok/s they carry

Guards against the traps that bit us in Lab 05
  - fresh random prompts every level, so prefix caching cannot inflate results (use --shared-prefix-words
    when you WANT cache hits)
  - no client-side connection cap: aiohttp's default of 100 silently capped our "c=128/256" runs
  - token counts come from the server's usage field; chunk/token mismatches are counted
  - the server's own peak running-request gauge (/metrics) is recorded, and the run warns when it is
    well below the load you asked for
"""
import argparse
import asyncio
import datetime
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time

import aiohttp
import numpy as np

WORDS = ("the of and to in is that for it as with was on be by at this from or an "
         "model token cache memory kernel batch latency throughput gpu tensor").split()


def make_prompt(n_words, rng):
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def parse_len(spec, rng):
    """'128' -> 128; '64-256' -> uniform integer in [64, 256]."""
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        return rng.randint(lo, hi)
    return int(spec)


async def one_request(session, url, model, prompt, max_tokens, track):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True}
    track["inflight"] += 1
    track["peak"] = max(track["peak"], track["inflight"])
    t0 = time.perf_counter()
    stamps, usage = [], None
    try:
        async with session.post(f"{url}/v1/completions", json=body) as resp:
            resp.raise_for_status()
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                chunk = json.loads(line[5:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if chunk.get("choices") and chunk["choices"][0].get("text"):
                    stamps.append(time.perf_counter())
    finally:
        track["inflight"] -= 1
    if not stamps:
        raise RuntimeError("no tokens received")
    n_chunks = len(stamps)
    n_out = usage["completion_tokens"] if usage else n_chunks
    ttft, e2e = stamps[0] - t0, stamps[-1] - t0
    return {"ttft": ttft, "e2e": e2e, "n_out": n_out, "n_in": usage["prompt_tokens"] if usage else None,
            "chunk_mismatch": n_chunks != n_out, "have_usage": usage is not None,
            "tpot": (e2e - ttft) / (n_out - 1) if n_out > 1 else None,
            "itl": list(np.diff(stamps))}


async def poll_server_metrics(session, url, stop, out):
    """Track the server's own view of load (vLLM Prometheus gauges). Silently gives up if unavailable."""
    while not stop.is_set():
        try:
            async with session.get(f"{url}/metrics") as r:
                if r.status != 200:
                    return
                text = await r.text()
        except Exception:
            return
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            try:
                val = float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                continue
            if line.startswith("vllm:num_requests_running"):
                out["running"] = max(out.get("running", 0), val)
            elif line.startswith("vllm:num_requests_waiting"):
                out["waiting"] = max(out.get("waiting", 0), val)
            elif line.startswith(("vllm:kv_cache_usage", "vllm:gpu_cache_usage")):
                out["kv"] = max(out.get("kv", 0), val)
        await asyncio.sleep(0.5)


class GpuSampler:
    """Samples nvidia-smi every 500 ms while a level runs (no-op on machines without a GPU)."""
    QUERY = "utilization.gpu,power.draw,memory.used,clocks.sm"

    def __init__(self, enabled):
        self.proc = None
        if enabled and shutil.which("nvidia-smi"):
            self.proc = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits", "-lms", "500"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    def stop(self):
        if not self.proc:
            return {}
        self.proc.terminate()
        out, _ = self.proc.communicate()
        cols = [[], [], [], []]
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                continue
            for c, p in zip(cols, parts):
                try:
                    c.append(float(p))
                except ValueError:
                    pass
        if not cols[0]:
            return {}
        return {"gpu_util_mean": round(float(np.mean(cols[0])), 1),
                "gpu_power_w_mean": round(float(np.mean(cols[1])), 1) if cols[1] else None,
                "gpu_mem_mib_peak": max(cols[2]) if cols[2] else None,
                "gpu_sm_mhz_mean": round(float(np.mean(cols[3]))) if cols[3] else None}


def pct(a, q):
    return round(float(np.percentile(a, q)) * 1000, 2) if len(a) else None


async def run_level(args, mode, level, n_requests):
    rng = random.Random(None if args.seed is None else args.seed * 1_000_003 + int(level * 1000))
    prefix = (make_prompt(args.shared_prefix_words, random.Random(12345)) + " ") if args.shared_prefix_words else ""
    prompts = [prefix + make_prompt(args.input_words, rng) for _ in range(n_requests)]
    out_lens = [parse_len(args.output_tokens, rng) for _ in range(n_requests)]
    results, errors = [], 0
    track = {"inflight": 0, "peak": 0}
    srv, stop = {}, asyncio.Event()
    # limit=0: no client-side connection cap (the default of 100 is a silent benchmark bug)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),
                                     timeout=aiohttp.ClientTimeout(total=1800)) as session:
        async def do(i):
            nonlocal errors
            try:
                results.append(await one_request(session, args.url, args.model, prompts[i], out_lens[i], track))
            except Exception as e:  # count failures, never hide them
                errors += 1
                if errors <= 3:
                    print("request failed:", repr(e), file=sys.stderr)

        poller = asyncio.create_task(poll_server_metrics(session, args.url, stop, srv))
        gpu = GpuSampler(not args.no_gpu_telemetry)
        t0 = time.perf_counter()
        if mode == "closed":
            sem = asyncio.Semaphore(int(level))

            async def gated(i):
                async with sem:
                    await do(i)
            await asyncio.gather(*(gated(i) for i in range(n_requests)))
        else:
            tasks, due = [], time.perf_counter()
            for i in range(n_requests):
                await asyncio.sleep(max(0.0, due - time.perf_counter()))
                tasks.append(asyncio.create_task(do(i)))
                due += rng.expovariate(level)
            await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
        stop.set()
        await poller
        gpu_stats = gpu.stop()
    return results, errors, wall, track, srv, gpu_stats


def summarize(args, mode, level, results, errors, wall, track, srv, gpu_stats, run_id):
    ttft = [r["ttft"] for r in results]
    e2e = [r["e2e"] for r in results]
    tpot = [r["tpot"] for r in results if r["tpot"] is not None]
    itl = [x for r in results for x in r["itl"]]
    out_tok = sum(r["n_out"] for r in results)
    in_tok = sum(r["n_in"] for r in results if r["n_in"] is not None)
    good = [r for r in results
            if (args.slo_ttft_ms is None or r["ttft"] * 1000 <= args.slo_ttft_ms)
            and (args.slo_tpot_ms is None or r["tpot"] is None or r["tpot"] * 1000 <= args.slo_tpot_ms)]
    row = {"run_id": run_id, "label": args.label, "mode": mode, "level": level,
           "input_words": args.input_words, "output_tokens": args.output_tokens,
           "shared_prefix_words": args.shared_prefix_words,
           "ok": len(results), "errors": errors, "wall_s": round(wall, 2),
           "req_per_s": round(len(results) / wall, 2), "out_tok_per_s": round(out_tok / wall, 1),
           "in_tok_per_s": round(in_tok / wall, 1) if in_tok else None,
           "mean_prompt_tokens": round(in_tok / max(1, sum(r["n_in"] is not None for r in results)), 1) if in_tok else None,
           "ttft_ms_p50": pct(ttft, 50), "ttft_ms_p99": pct(ttft, 99),
           "tpot_ms_p50": pct(tpot, 50), "tpot_ms_p99": pct(tpot, 99),
           "itl_ms_p50": pct(itl, 50), "itl_ms_p99": pct(itl, 99),
           "e2e_ms_p50": pct(e2e, 50), "e2e_ms_p99": pct(e2e, 99),
           "chunk_mismatch_reqs": sum(r["chunk_mismatch"] for r in results),
           "usage_missing_reqs": sum(not r["have_usage"] for r in results),
           "client_peak_inflight": track["peak"],
           "server_peak_running": srv.get("running"), "server_peak_waiting": srv.get("waiting"),
           "server_peak_kv_usage": srv.get("kv")}
    if args.slo_ttft_ms is not None or args.slo_tpot_ms is not None:
        row["goodput_frac"] = round(len(good) / max(1, len(results)), 3)
        row["goodput_tok_per_s"] = round(sum(r["n_out"] for r in good) / wall, 1)
    row.update(gpu_stats)
    return row


def check_row(row, mode, level):
    """Print loud warnings when the run did not measure what was asked."""
    if row["errors"]:
        print(f"WARNING level {level}: {row['errors']} requests failed", file=sys.stderr)
    if row["usage_missing_reqs"]:
        print(f"WARNING level {level}: server sent no usage; token counts are chunk counts", file=sys.stderr)
    if mode == "closed" and row["server_peak_running"] is not None and row["server_peak_running"] < 0.9 * level:
        print(f"WARNING level {level}: server saw at most {row['server_peak_running']:.0f} requests running, "
              f"fewer than the {level} requested (client cap? scheduler limit such as --max-num-seqs?). "
              f"Treat this level as the lower number.", file=sys.stderr)


def environment(args, run_id):
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip() or None
        except Exception:
            return None
    meta = {"run_id": run_id, "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "argv": sys.argv[1:], "label": args.label, "python": platform.python_version(),
            "git_commit": sh(["git", "rev-parse", "--short", "HEAD"]),
            "gpu": sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
            "server_flags_note": args.server_flags}
    try:
        import urllib.request
        meta["server_version"] = json.load(urllib.request.urlopen(f"{args.url}/version", timeout=5))
    except Exception:
        meta["server_version"] = None
    return meta


async def warmup(args):
    rng = random.Random()
    track = {"inflight": 0, "peak": 0}
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as s:
        await asyncio.gather(*(one_request(s, args.url, args.model, make_prompt(args.input_words, rng), 32, track)
                               for _ in range(args.warmup)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="mock")
    ap.add_argument("--num-requests", type=int, default=64, help="requests per level")
    ap.add_argument("--concurrency", type=int, nargs="+", help="closed-loop levels (users in flight)")
    ap.add_argument("--rate", type=float, nargs="+", help="open-loop levels (requests/s, Poisson arrivals)")
    ap.add_argument("--input-words", type=int, default=256, help="~prompt length in words (about 1.2-1.3 tokens each)")
    ap.add_argument("--shared-prefix-words", type=int, default=0,
                    help="identical prefix on every request (prefix-cache and KV-reuse experiments)")
    ap.add_argument("--output-tokens", default="128", help="fixed ('128') or uniform range ('64-256')")
    ap.add_argument("--slo-ttft-ms", type=float, help="goodput: max TTFT")
    ap.add_argument("--slo-tpot-ms", type=float, help="goodput: max TPOT")
    ap.add_argument("--warmup", type=int, default=8, help="throwaway requests before measuring (0 to skip)")
    ap.add_argument("--no-gpu-telemetry", action="store_true")
    ap.add_argument("--label", default="", help="config name stored in each row, e.g. 'fp8' or 'chunked-prefill'")
    ap.add_argument("--server-flags", default="", help="paste the server launch flags; stored in the meta file")
    ap.add_argument("--seed", type=int, default=None, help="omit for fresh random prompts (recommended)")
    ap.add_argument("--out", default="results/run.jsonl")
    args = ap.parse_args()

    if args.rate and args.concurrency:
        ap.error("choose --concurrency (closed loop) or --rate (open loop), not both")
    mode, levels = ("open", args.rate) if args.rate else ("closed", args.concurrency or [1, 4, 16])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    meta = environment(args, run_id)
    with open(os.path.splitext(args.out)[0] + ".meta.jsonl", "a") as f:
        f.write(json.dumps(meta) + "\n")
    if args.warmup:
        asyncio.run(warmup(args))

    for level in levels:
        results, errors, wall, track, srv, gpu_stats = asyncio.run(run_level(args, mode, level, args.num_requests))
        if not results:
            print(f"level {level}: every request failed", file=sys.stderr)
            continue
        row = summarize(args, mode, level, results, errors, wall, track, srv, gpu_stats, run_id)
        check_row(row, mode, level)
        print(json.dumps(row))
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()

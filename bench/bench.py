"""Load generator + metrics for any OpenAI-compatible streaming endpoint (vLLM, SGLang, mock).

Closed-loop: keeps exactly `--concurrency` requests in flight until `--num-requests` finish.
Each streamed chunk is treated as one output token, which holds for vLLM/SGLang defaults.

Metrics (per request, then aggregated):
  TTFT  time to first token      -> dominated by queueing + prefill
  ITL   inter-token latency      -> decode step time as seen by one user
  E2E   total request latency
  Throughput = total output tokens / wall-clock time  -> what you pay for per GPU
"""
import argparse
import asyncio
import json
import random
import time

import aiohttp
import numpy as np

WORDS = ("the of and to in is that for it as with was on be by at this from or an "
         "model token cache memory kernel batch latency throughput gpu tensor").split()


def make_prompt(n_words: int, rng: random.Random) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


async def one_request(session, url, model, prompt, max_tokens):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "ignore_eos": True}
    t0 = time.perf_counter()
    stamps = []
    async with session.post(f"{url}/v1/completions", json=body) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[5:])
            if chunk.get("choices") and chunk["choices"][0].get("text"):
                stamps.append(time.perf_counter())
    if not stamps:
        raise RuntimeError("no tokens received")
    return {"ttft": stamps[0] - t0, "e2e": stamps[-1] - t0, "n_out": len(stamps),
            "itl": list(np.diff(stamps))}


async def run(args):
    # Fresh prompts per run and per concurrency level by default. A fixed seed would repeat prompts, and
    # vLLM's prefix caching would then skip their prefill and inflate throughput. Pass --seed only to
    # reproduce a run exactly (and then expect cache hits on repeats).
    rng = random.Random(None if args.seed is None else args.seed * 1_000_003 + args.concurrency)
    prompts = [make_prompt(args.input_words, rng) for _ in range(args.num_requests)]
    sem = asyncio.Semaphore(args.concurrency)
    results, errors = [], 0
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def worker(p):
            nonlocal errors
            async with sem:
                try:
                    results.append(await one_request(session, args.url, args.model, p, args.output_tokens))
                except Exception as e:  # count failures, don't hide them
                    errors += 1
                    print("request failed:", repr(e))
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(p) for p in prompts))
        wall = time.perf_counter() - t0
    return results, errors, wall


def pct(a, q):
    return float(np.percentile(a, q)) * 1000 if len(a) else float("nan")  # ms


def summarize(args, results, errors, wall):
    ttft = [r["ttft"] for r in results]
    e2e = [r["e2e"] for r in results]
    itl = [x for r in results for x in r["itl"]]
    total_out = sum(r["n_out"] for r in results)
    return {
        "concurrency": args.concurrency, "input_words": args.input_words,
        "output_tokens": args.output_tokens, "ok": len(results), "errors": errors,
        "wall_s": round(wall, 2), "req_per_s": round(len(results) / wall, 2),
        "out_tok_per_s": round(total_out / wall, 1),
        "ttft_ms_p50": round(pct(ttft, 50), 1), "ttft_ms_p99": round(pct(ttft, 99), 1),
        "itl_ms_p50": round(pct(itl, 50), 2), "itl_ms_p99": round(pct(itl, 99), 2),
        "e2e_ms_p50": round(pct(e2e, 50), 1), "e2e_ms_p99": round(pct(e2e, 99), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="mock")
    ap.add_argument("--num-requests", type=int, default=64)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16],
                    help="one or more levels; each runs a full sweep point")
    ap.add_argument("--input-words", type=int, default=256, help="~prompt length (words ~ 1.3 tokens)")
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=None, help="omit for fresh random prompts (recommended)")
    ap.add_argument("--out", default="results/run.jsonl")
    args = ap.parse_args()

    rows = []
    for c in args.concurrency:
        args.concurrency_cur = c
        a = argparse.Namespace(**{**vars(args), "concurrency": c})
        results, errors, wall = asyncio.run(run(a))
        row = summarize(a, results, errors, wall)
        rows.append(row)
        print(json.dumps(row))
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

"""Lab 09: a discrete-event simulator of a continuous-batching LLM engine.

Each engine step takes   step_ms = base_ms + per_seq_ms * (decoding_seqs - 1) + prefill_ms_per_token * prefill_tokens
and the scheduler decides, every step, which waiting requests join, which prefill chunks run, and who decodes.
It models the things Lab 05 and Lab 08 found to matter (queueing, prefill/decode interference, token budget, KV
capacity) and nothing else, so it is cheap to run and easy to reason about. Calibrate it, then test it on points it was
not fitted to before trusting any policy comparison.

    python simulator.py validate                 # closed loop vs Lab 05 measurements
    python simulator.py chunking                 # chunked prefill on/off, mixed prompt lengths
    python simulator.py policies --rates 16 20 30   # FCFS vs shortest-prompt-first vs priority, open loop
"""
import argparse
import heapq
import itertools
import random
import statistics
from dataclasses import dataclass, field

# Lab 05 clean results (H100, Llama-3.1-8B, ~300-token prompts, 128 output tokens)
LAB05 = {1: 160.3, 8: 1156.7, 16: 2121.3, 32: 3676.0, 64: 5866.0, 128: 7056.0}  # tokens/s; c=128 really ~100 in flight


@dataclass
class Model:
    base_ms: float = 6.05             # decode step with one sequence (Lab 05, 1 user)
    per_seq_ms: float = 0.0244        # extra per additional decoding sequence (fit: 8.47 ms at ~100 sequences)
    prefill_ms_per_token: float = 0.045  # fitted to the 64-user throughput (see `validate`)
    max_seqs: int = 256
    token_budget: int = 8192          # tokens per step (assumed; check the engine's actual value)
    kv_capacity_tokens: int = 438_000  # Lab 05 server log
    chunked_prefill: bool = True


@dataclass(order=True)
class Req:
    arrival: float
    rid: int = field(compare=False)
    prompt: int = field(compare=False)
    out: int = field(compare=False)
    prio: int = field(compare=False, default=0)
    prefill_left: int = field(compare=False, default=0)
    out_left: int = field(compare=False, default=0)
    first_token: float = field(compare=False, default=None)
    finish: float = field(compare=False, default=None)

    def __post_init__(self):
        self.prefill_left, self.out_left = self.prompt, self.out


# ---- policies: the order in which waiting requests are admitted ---------------------------------------------
POLICIES = {
    "fcfs": lambda r: (r.arrival,),
    "sjf_prompt": lambda r: (r.prompt, r.arrival),          # shortest prompt first (needs only the prompt length)
    "sjf_oracle": lambda r: (r.prompt + r.out, r.arrival),  # needs the output length: an upper bound on what is possible
    "priority": lambda r: (r.prio, r.arrival),              # interactive class first, FCFS inside a class
}


def run(model, policy, initial, on_finish=None):
    """Simulate until every request finishes. `initial` is a list of Req; `on_finish(req, now)` may return new Reqs."""
    key = POLICIES[policy]
    pending = list(initial)
    heapq.heapify(pending)
    waiting, running, done = [], [], []
    t = 0.0
    while pending or waiting or running:
        while pending and pending[0].arrival <= t:
            waiting.append(heapq.heappop(pending))
        if not waiting and not running:
            t = pending[0].arrival
            continue
        # admission, in policy order, limited by sequence slots and KV space (no skipping: keeps the order honest)
        kv_used = sum(r.prompt + (r.out - r.out_left) for r in running)
        for r in sorted(waiting, key=key):
            if len(running) >= model.max_seqs or kv_used + r.prompt > model.kv_capacity_tokens:
                break
            running.append(r)
            waiting.remove(r)
            kv_used += r.prompt
        decoders = [r for r in running if r.prefill_left == 0]
        prefillers = [r for r in running if r.prefill_left > 0]
        take = {}
        if model.chunked_prefill:
            budget = model.token_budget - len(decoders)
            for r in prefillers:
                n = min(r.prefill_left, max(budget, 0))
                if n:
                    take[r.rid] = n
                    budget -= n
        else:  # classic scheduling: a step is prefill-only if any prompt is waiting, so decoders stall
            if prefillers:
                budget = model.token_budget
                for r in prefillers:
                    if budget <= 0 and take:
                        break
                    take[r.rid] = r.prefill_left
                    budget -= r.prefill_left
                decoders = []
        n_dec = len(decoders)
        step_ms = (model.base_ms + model.per_seq_ms * max(n_dec - 1, 0)
                   + model.prefill_ms_per_token * sum(take.values()))
        t += step_ms / 1000
        finished = []
        for r in decoders:
            r.out_left -= 1
            if r.out_left == 0:
                finished.append(r)
        for r in prefillers:
            n = take.get(r.rid, 0)
            r.prefill_left -= n
            if n and r.prefill_left == 0:   # prefill complete: the first token comes out of this step
                r.first_token = t
                r.out_left -= 1
                if r.out_left == 0:
                    finished.append(r)
        for r in finished:
            r.finish = t
            running.remove(r)
            done.append(r)
            if on_finish:
                for nr in on_finish(r, t) or []:
                    heapq.heappush(pending, nr)
    return done, t


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q / 100 * len(xs)))] * 1000 if xs else float("nan")


def summarize(done, wall, label=""):
    ttft = [r.first_token - r.arrival for r in done]
    e2e = [r.finish - r.arrival for r in done]
    tpot = [(r.finish - r.first_token) / (r.out - 1) for r in done if r.out > 1]
    return {"label": label, "n": len(done), "tok_s": round(sum(r.out for r in done) / wall, 1),
            "ttft_p50": round(pct(ttft, 50), 1), "ttft_p99": round(pct(ttft, 99), 1),
            "tpot_p50": round(pct(tpot, 50), 2), "tpot_p99": round(pct(tpot, 99), 2),
            "e2e_p50": round(pct(e2e, 50)), "e2e_p99": round(pct(e2e, 99))}


# ---- workloads ---------------------------------------------------------------------------------------------
def closed_loop(model, policy, users, n_requests, prompt=300, out=128, seed=0):
    ids = itertools.count()
    issued = [users]

    def spawn(now):
        return Req(now, next(ids), prompt, out)

    def on_finish(r, now):
        if issued[0] < n_requests:
            issued[0] += 1
            return [spawn(now)]
        return []
    done, wall = run(model, policy, [spawn(0.0) for _ in range(users)], on_finish)
    return done, wall


def open_loop_mixed(model, policy, rate, n_requests, seed=1, batch_share=0.2):
    rng = random.Random(seed)
    t, reqs = 0.0, []
    for i in range(n_requests):
        t += rng.expovariate(rate)
        if rng.random() < batch_share:   # long-prompt batch job, low priority
            reqs.append(Req(t, i, int(rng.lognormvariate(8.0, 0.4)), int(rng.lognormvariate(5.9, 0.5)) + 1, prio=1))
        else:                              # interactive chat, high priority
            reqs.append(Req(t, i, int(rng.lognormvariate(5.7, 0.5)) + 1, int(rng.lognormvariate(4.6, 0.6)) + 1, prio=0))
    return run(model, policy, reqs)


# ---- commands ----------------------------------------------------------------------------------------------
def cmd_validate(a):
    print("Calibration: base_ms from 1 user, per_seq_ms from ITL at ~100 users, prefill cost fitted to 64-user throughput.")
    best = None
    for p in [x / 1000 for x in range(10, 121, 5)]:
        m = Model(prefill_ms_per_token=p)
        done, wall = closed_loop(m, "fcfs", 64, 640)
        err = abs(summarize(done, wall)["tok_s"] - LAB05[64])
        if best is None or err < best[0]:
            best = (err, p)
    p = best[1]
    print(f"fitted prefill_ms_per_token = {p}\n")
    print(f"{'users':>6} {'measured tok/s':>15} {'simulated tok/s':>16} {'error':>7}   (fitted on 1 and 64; the others are held out)")
    for users, meas in LAB05.items():
        u = 100 if users == 128 else users   # Lab 05's "128" really had ~100 in flight (client cap)
        done, wall = closed_loop(Model(prefill_ms_per_token=p), "fcfs", u, max(10 * u, 200))
        sim = summarize(done, wall)["tok_s"]
        tag = "  (fitted)" if users in (1, 64) else ""
        print(f"{u:>6} {meas:>15.0f} {sim:>16.0f} {100 * (sim - meas) / meas:>+6.1f}%{tag}")


def cmd_chunking(a):
    print("Open loop, mixed workload (80% chat, 20% long-prompt batch), FCFS, token budget 2048\n")
    rows = []
    for chunked in (False, True):
        m = Model(chunked_prefill=chunked, token_budget=2048)
        done, wall = open_loop_mixed(m, "fcfs", a.rate, a.n)
        chat = [r for r in done if r.prio == 0]
        s = summarize(done, wall, f"chunked={chunked}")
        s["chat_tpot_p99"] = round(pct([(r.finish - r.first_token) / (r.out - 1) for r in chat if r.out > 1], 99), 2)
        rows.append(s)
    for s in rows:
        print(s)


def cmd_policies(a):
    for rate in a.rates:
        print(f"\narrival rate {rate}/s")
        for pol in POLICIES:
            done, wall = open_loop_mixed(Model(token_budget=2048), pol, rate, a.n)
            chat = [r for r in done if r.prio == 0]
            batch = [r for r in done if r.prio == 1]
            s = summarize(done, wall, pol)
            print(f"  {pol:<11} tok/s {s['tok_s']:>7}  chat TTFT p50/p99 {pct([r.first_token - r.arrival for r in chat], 50):>7.0f}/"
                  f"{pct([r.first_token - r.arrival for r in chat], 99):>7.0f} ms   batch E2E p99 {pct([r.finish - r.arrival for r in batch], 99):>8.0f} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate").set_defaults(fn=cmd_validate)
    c = sub.add_parser("chunking")
    c.add_argument("--rate", type=float, default=12)
    c.add_argument("--n", type=int, default=3000)
    c.set_defaults(fn=cmd_chunking)
    p = sub.add_parser("policies")
    p.add_argument("--rates", type=float, nargs="+", default=[16, 20, 30])
    p.add_argument("--n", type=int, default=3000)
    p.set_defaults(fn=cmd_policies)
    args = ap.parse_args()
    args.fn(args)

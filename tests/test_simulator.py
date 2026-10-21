"""Self-tests for the Lab 09 scheduler simulator. No GPU needed.

    python tests/test_simulator.py
Each test is checked against something computable by hand.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "labs", "09_custom_scheduling"))
import simulator as s  # noqa: E402


def test_single_user_matches_hand_calculation():
    m = s.Model(prefill_ms_per_token=0.02)
    done, _ = s.closed_loop(m, "fcfs", 1, 5, prompt=300, out=128)
    r = done[0]
    ttft_ms = (r.first_token - r.arrival) * 1000
    e2e_ms = (r.finish - r.arrival) * 1000
    assert abs(ttft_ms - (6.05 + 300 * 0.02)) < 0.01, ttft_ms                  # one prefill step
    assert abs(e2e_ms - (ttft_ms + 127 * 6.05)) < 0.5, e2e_ms                 # then 127 single-sequence decode steps


def test_every_request_finishes_in_order_of_events():
    done, wall = s.open_loop_mixed(s.Model(token_budget=2048), "fcfs", 10, 400)
    assert len(done) == 400
    assert all(r.arrival <= r.first_token <= r.finish <= wall + 1e-9 for r in done)
    assert all(r.out_left == 0 and r.prefill_left == 0 for r in done)


def test_priority_protects_interactive_traffic_under_overload():
    chat_p99 = {}
    for pol in ("fcfs", "priority"):
        done, _ = s.open_loop_mixed(s.Model(token_budget=2048), pol, 30, 1500)
        chat = [r.first_token - r.arrival for r in done if r.prio == 0]
        chat_p99[pol] = s.pct(chat, 99)
    assert chat_p99["priority"] < chat_p99["fcfs"] / 2, chat_p99


def test_policy_is_irrelevant_without_queueing():
    outs = {}
    for pol in ("fcfs", "priority"):
        done, wall = s.open_loop_mixed(s.Model(token_budget=2048), pol, 4, 600)
        outs[pol] = s.summarize(done, wall)["ttft_p50"]
    assert abs(outs["fcfs"] - outs["priority"]) < 5, outs  # ms


def test_kv_capacity_limits_throughput():
    big, wall_big = s.closed_loop(s.Model(), "fcfs", 16, 160)
    small, wall_small = s.closed_loop(s.Model(kv_capacity_tokens=1000), "fcfs", 16, 160)   # about 3 prompts fit
    assert s.summarize(small, wall_small)["tok_s"] < 0.5 * s.summarize(big, wall_big)["tok_s"]


def test_chunked_prefill_reduces_decode_stalls():
    tp = {}
    for chunked in (False, True):
        done, _ = s.open_loop_mixed(s.Model(chunked_prefill=chunked, token_budget=2048), "fcfs", 12, 1500)
        chat = [r for r in done if r.prio == 0 and r.out > 1]
        tp[chunked] = s.pct([(r.finish - r.first_token) / (r.out - 1) for r in chat], 99)
    assert tp[True] < tp[False], tp


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_")):
        try:
            fn()
            print("PASS", name)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL", name, "->", repr(e))
    sys.exit(1 if failed else 0)

# Real chatbot screen: glm-5.3-flash on 16 H100, BLIS v0.14.0

Stack: inference-sim v0.14.0 @ f2ba6bb (kernel v0.4.0, blis-vllm v0.32.0), blis-catalog 0.3.0,
blis-registry v0.2.0, `engine_version: "0.29.0"`, 1280 requests per point. This is the first
screen run after PR #6, so the workload is really the chatbot preset (256/256, max 800/1024), not
BLIS's 512/512 defaults. A second real-workload scenario was screened alongside it; it is reserved as
the clean-room test of LLM-designed sweeps, so its screen is not published until that test has run.

| | chatbot |
|---|---|
| screening space | `spaces/screen/glm53flash-h100-chatbot-16-real-screen.yaml` |
| SLO | ITL p99 ≤ 20 ms |
| records (local, `results/` is not committed) | `results/screen-chatbot/sensitivity.jsonl` (182) |
| wall time (4 workers, M5 Max, alongside the other screen) | 7 min |

It uses S-mid v3's knobs and value lists, at six bases. Defaults: EP
on, internal DP LB, load-only router, `max_num_seqs` 256, fp8 KV cache, gmu 0.9, mnbt 8192, block 64.
"inflight ext" = inflight-load router with external DP LB. Each cell is the one-axis spread
(max − min)/max in %, over the scored values.

## Chatbot, ITL p99 ≤ 20 ms
| base | tok/s/GPU (C*) | EP | DPLB | kvdt | mns | gmu | mnbt | blk | sched | preempt | EPP | qw | kvw | pw | aw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tp8·dp1 | 42.0 (7) | 17.3 | – | 0.1 | 0.7 | 0.0 | 0.0 | 0.3 | 0.0 | 0.0 | 0.5 | 0.0 | 18.9 | – | – |
| tp8·dp2 | 96.2 (24) | 86.7 | 10.8 | 3.0 | 83.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | – | – |
| tp16·dp1 | 77.6 (16) | 70.3 | – | 0.1 | 73.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | – | – |
| tp4·dp2 | 26.4 (6) | 15.4 | 2.9 | 0.1 | 0.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 19.2 | 0.0 | 0.0 | – | – |
| tp2·dp4 | 20.4 (6) | 9.3 | 5.5 | 0.0 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 20.1 | 0.0 | 0.0 | – | – |
| tp8·dp2 inflight ext | 96.1 (24) | 86.9 | 0.1 | 0.1 | 83.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 10.7 | 0.0 | 0.0 | 0.0 | 10.4 |

- **The topology ranking reverses from the 512/512 scenario.** 1 × tp8·dp2 + EP (96.2 tok/s/GPU, C* = 24)
  and 1 × tp16 + EP (77.6) beat 2 × tp8 + EP (42.0, C* = 7), which was S-mid v3's optimum. Chatbot's
  ITL p99 jumps from 8.9 ms at c = 7 to 22.4 ms at c = 8 on 2 × tp8 + EP, on all four workload seeds
  checked (0–3, within 0.1 ms), while p90 and p95 stay at about 8.4 ms. So C* sits on a deterministic
  cliff, and the one-instance shapes reach a higher one.
- **`max_num_seqs` interacts with topology:** 84% at tp8·dp2 and 74% at tp16, where values below 256
  collapse C* to 3–6, but under 1% at 2 × tp8, tp4·dp2 and tp2·dp4.
- **EP** is worth 9–87%, most at the one-instance shapes (without it tp8·dp2 falls to 12.8).
- **Router family** is worth 19–20% at tp4·dp2 and tp2·dp4 (inflight-load best), 11% at tp8·dp2
  with external LB, and 0–0.5% elsewhere. The active-request weight at 0 costs 10% there.
- **The kv-cache-utilization weight at 0** costs 19% at 2 × tp8 (as in the 512/512 scenarios: no
  balancing scorer left).
- **DP LB:** external costs 11% at tp8·dp2 under load-only, 0.1% under inflight-load.
- `cache_dtype` 3% at tp8·dp2; otherwise under 1%.
- **Flat (under 1% at every base):** `gpu_memory_utilization`, `max_num_batched_tokens`,
  `block_size`, `scheduler`, `preemption_policy`, queue weight, prefix-cache weight.
- Every probe was ITL-bound; no TTFT SLO was set.

## What this means for the benchmark space

S-chat (`spaces/glm53flash-h100-chatbot-real-16.yaml`) has what S-mid v3 and S-mn lacked: an engine
knob whose effect depends on the topology, and an optimum away from the default. The interaction is
topology × `max_num_seqs`, with EP and router as second-order axes. Pinned: gmu, mnbt, block size,
scheduler, preemption, queue and prefix weights. `cache_dtype` (3%) is borderline and kept. The screen
varies one knob at a time around six bases, so the grid is still the ground truth on pairwise
interactions.

## Ground truth (full grid, 2026-10-31)

> **Caveat (2026-11-01).** This space pins `cudagraph_mode` to PIECEWISE. With it free, the optimum is
> at least 108.9 (FULL), 13% higher; see `docs/experiments/s-chat-large.md`. The tables below are for
> the pinned space.

`uv run cf-grid spaces/glm53flash-h100-chatbot-real-16.yaml --cache cache --out results/s-chat/grid.jsonl --workers 15`:
7,680 configs, all scored, 40,310 unique BLIS runs, 3 h 25 min on the M5 Max.

- **Optimum: 96.2 tok/s per GPU at 1 × tp8·dp2 + EP** (internal DP LB, `max_num_seqs` 256, load-only
  router, kv weight 1). All 96 configs within 1% are that topology.
- **The default (2 × tp8 + EP) is 56% below it** at 42.0. In S-mid v3 the default was 0.11% off.

| topology | best tok/s/GPU | vs optimum |
|---|---|---|
| 1 × tp8·dp2 + EP | 96.2 | — |
| 1 × tp16 + EP | 77.6 | −19% |
| 2 × tp8 + EP | 42.2 | −56% |
| 2 × tp8 | 34.8 | −64% |
| 2 × tp4·dp2 + EP | 32.8 | −66% |
| 2 × tp4·dp2 | 27.7 | −71% |
| 1 × tp4·dp4 + EP | 26.0 | −73% |
| 2 × tp2·dp4 + EP | 25.5 | −74% |
| every other topology | ≤ 23.1 | −76% or worse |

Best score per `max_num_seqs` value:

| topology | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| 1 × tp8·dp2 + EP | 15.9 | 15.9 | 29.5 | 96.2 | 96.2 |
| 1 × tp16 + EP | 20.4 | 20.4 | 36.4 | 77.6 | 77.6 |
| 2 × tp8 + EP | 41.9 | 41.9 | 42.0 | 42.2 | 42.2 |
| 2 × tp8 | 34.6 | 34.6 | 34.7 | 34.8 | 34.8 |

So the two best topologies are good only with `max_num_seqs` ≥ 256: below it they fall to or under the
2 × tp8 shapes. A search that samples a low `max_num_seqs` with them misjudges them as weak.

## Benchmark (budget 1000, 10 seeds)

`uv run cf-bench spaces/glm53flash-h100-chatbot-real-16.yaml --cache cache --out results/s-chat/runs --budget 1000 --seeds 0-9 --jobs 4 --workers 4 --methods bsa,bsa-feas,bsa-diag,random,tpe,smac,incumbent,coord-ascent`,
then `cf-analyze` against the grid. 1.7 min: every probe was a cache hit from the grid.

| method | regret at B=100 | B=200 | B=1000 | runs to within 1% | seeds within 0.5% at B=1000 |
|---|---|---|---|---|---|
| tpe | 18.7 [0.13, 18.7] | 0.11 [0, 18.7] | 0 [0, 0] | **129** [76, 256] | 100% |
| incumbent (deterministic) | 0 | 0 | 0 | 91 | 100% |
| random | 18.7 [4.67, 18.7] | 6.5 [0.03, 16.6] | 0 [0, 2.17] | 254 [104, 876] (3 censored) | 70% |
| smac | 36.5 [18.7, 59.7] | 18.7 [12.6, 18.7] | 0 [0, 0] | 554 [334, 860] (2 censored) | 80% |
| bsa | 18.7 [18.7, 65.1] | 18.7 [18.7, 61.2] | 0 [0, 14] | 573 [414, 925] (3 censored) | 70% |
| bsa-feas | 18.7 [18.7, 18.7] | 18.7 [4.67, 18.7] | 0.09 [0, 14.7] | 507 [173, 1001] (4 censored) | 60% |
| bsa-diag | 18.7 [18.7, 45.2] | 18.7 [4.67, 45.2] | 1.43 [0, 18.7] | 764 [208, 1001] (5 censored) | 50% |
| coord-ascent | 18.7 [4.67, 18.7] | 18.7 [4.67, 18.7] | 18.7 [4.67, 18.7] | 1001 [285, 1001] (7 censored) | 30% |

Regret in tok/s per GPU, median [IQR] over seeds. A censored seed never got within 1%.

- **TPE is the clear winner.** It reaches within 1% in a median 129 runs on every seed. BSA needs 573
  (p = 0.002, paired Wilcoxon) and misses on 3 of 10 seeds.
- **The incumbent reaches the optimum in 91 runs.** Its sequence happens to try 1 × tp8·dp2 early.
- **The trap is 1 × tp16 + EP (77.6, regret 18.7).** From there no single change leads to the optimum:
  raising dp makes the width 32, over the budget. Coordinate ascent moves to tp16 first and stays there
  on 7 of 10 seeds. BSA's sampling distribution concentrates on tp16 the same way on the censored seeds.
- **BSA's biasers do not help here.** bsa-feas and bsa-diag are no better than plain bsa (p ≥ 0.56).
- **BSA is behind random search** on runs to 1% (573 vs 254 median), though not significantly (p = 0.38).
- So S-chat has interacting knobs, but its hard part is one topology trap, and TPE handles that best.

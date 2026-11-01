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
> at least 108.9 (FULL), 13% higher; see "`cudagraph_mode` on chatbot" below. The tables below are for
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

## `cudagraph_mode` on chatbot

S-chat pins `cudagraph_mode` to PIECEWISE, the runner's default. The S-mid v3 screen had not screened it (its
"values accepted" check only ran each value once). A screen with `cudagraph_mode` and
`max_model_len` added (`spaces/screen/glm53flash-h100-chatbot-16-real-screen.yaml`, 9 bases, 323
probes, 3 min):

| base | NONE | PIECEWISE | FULL | FULL_DECODE_ONLY | FULL_AND_PIECEWISE |
|---|---|---|---|---|---|
| 1 × tp8·dp2 + EP | 8.6 | 96.2 | **105.8** | 16.7 | 99.2 |
| same, inflight router + external LB | 8.5 | 96.1 | **108.9** | 16.3 | 99.2 |
| 1 × tp16 + EP | 9.9 | 77.6 | **88.3** | 21.9 | 80.8 |
| 2 × tp8 + EP | 22.4 | 42.0 | 44.4 | 44.1 | **44.5** |
| 2 × tp4·dp2 + EP | 15.4 | 26.4 | **27.6** | 27.4 | 27.5 |
| 2 × tp2·dp4 + EP | 9.9 | 20.4 | **21.1** | 20.9 | **21.1** |

- FULL is best or within 0.3% of best at every base, worth +10–14% at the one-instance shapes.
- FULL_DECODE_ONLY is a second topology-dependent trap, like low `max_num_seqs`: near-best at the
  two-instance shapes, −72% to −83% at the one-instance ones (C* falls to 3).
- NONE costs 47–92% everywhere. `max_model_len` is flat (0.0%) at every base.
- The other knobs' spreads are as in the earlier screen, with FULL as well as PIECEWISE.

**Caveat on S-chat (decided with the user, 2026-11-01).** S-chat's grid ran with `cudagraph_mode`
pinned to PIECEWISE, so 96.2 is the optimum of that pinned space, not of the knob space; with
`cudagraph_mode` free it is at least 108.9 (FULL at 1 × tp8·dp2 + EP with the inflight-load router and
external DP LB). S-chat is not regridded.

## S-chat-large

`spaces/glm53flash-h100-chatbot-real-16-large.yaml`: S-large v3's knobs on chatbot, 707,788,800 feasible
configs (the same count as S-large v3; the raw product of the axes, 9,953,280,000, also counts infeasible
topologies and labels on gated-off knobs). S-chat is a sub-space of it with the same defaults (checked by
`tests/test_spaces.py`). No grid.

# S-mid: glm-5.3-flash on 16 H100, chatbot (real BLIS)

Two versions. **S-mid v1** (sections up to "Incumbent") is the first space, with the brief's axes.
**S-mid v2** (from "Enrichment" on) is the committed `spaces/glm53flash-h100-chatbot-16.yaml`.
The v1 YAML is `git show 40d4910:spaces/glm53flash-h100-chatbot-16.yaml`.

Space: `spaces/glm53flash-h100-chatbot-16.yaml`. Scenario: glm-5.3-flash (MoE, 288 routed experts,
64 KV heads, catalog 0.2.1), h100, chatbot, GPU budget 16, ITL p99 ≤ 20 ms, headroom 0.8, seed 0,
engine_version 0.29.0. Precision follows upstream's glm run: `quantization: fp8` (single-value knob)
and `cache_dtype` ∈ {auto, fp8}. Score = −tokens/s per GPU at c_use = ⌊0.8·C*⌋.

Pinned stack: inference-sim v0.10.0 @ 0d092d4ef64f605046de1181adf8e2b6ee564725 (kernel v0.1.0),
blis-catalog 0.2.1 @ 28e82d4, blis-registry v0.1.1 @ f7519b1 (`scripts/setup_blis.sh`).

Defaults: tp 8, dp 1, EP true, max_num_seqs 256, max_num_batched_tokens 8192,
gpu_memory_utilization 0.9, cache_dtype fp8, routing_policy round-robin (num_instances = 16/(tp·dp) = 2).
The runner fills `block_size` 64 (it treats glm-5.3 as a DSA model) and the other engine defaults
(max_model_len 32768, cudagraph PIECEWISE).

## Size

| | combinations | feasible (topology rules) |
|---|---|---|
| brief's axes (max_num_seqs 5 values) | 19,200 | 11,400 |
| **S-mid v1** (max_num_seqs {32, 256}) | 7,680 | **4,560** |

19 topologies pass the rules (tp·dp ≤ 8, divides 16, 64 % tp = 0, EP needs width > 1), × 240 engine
combinations.

**Only 8 of the 19 topologies run in BLIS**: the kernel's memory model fits the weights only at
width tp·dp = 8: (tp, dp) ∈ {(8,1), (4,2), (2,4), (1,8)}, each with EP on or off. Every width-1/2/4
topology fails on the first run (c = 1) with "no room for KV" (tp4·dp1: 79.8 GiB fixed per rank vs a
72 GiB budget) or an impossible per-rank occupancy (tp1: 304 GiB, tp2·dp1: 155 GiB, tp1·dp4: 92 GiB,
tp2·dp2: 84 GiB). 1,920 of the 4,560 v1 configs are width 8, and **1,848 score**: the other 72 width-8 configs are
tp1·dp8 at gpu_memory_utilization 0.7, which also fails. The other 2,640 score None at one run each.
This floor is now explicit: `model_entry.min_footprint: 8` makes `is_feasible` reject tp·dp < 8 (both
spaces).

## Sensitivity (one axis at a time, real BLIS)

`scripts/sensitivity.py SPACE --cache cache/ --base '{...}'`. Spread = (max − min)/max of tps/GPU over the
axis's values; C* in parentheses; `*` = base value. Run with the brief's full value lists (before the
max_num_seqs trim).

**From the default (tp8·dp1, EP, round-robin): 56.3 tps/GPU, C* = 13.**

| axis | values: tps/GPU (C*) | spread |
|---|---|---|
| tp | 2: fail, 4: fail, 8*: 56.3 (13) | (topology) |
| dp | 1* only feasible at tp 8 | (topology) |
| enable_expert_parallel | false 52.2 (13), true* 56.3 (13) | 7.4% |
| cache_dtype | auto 56.0, fp8* 56.3 | **0.5% (<1%)** |
| max_num_seqs | 32 … 512: all 56.3 (13) | **0.0% (<1%)** |
| max_num_batched_tokens | 2048 … 32768: all 56.3 (13) | **0.0% (<1%)** |
| gpu_memory_utilization | 0.7 … 0.95: all 56.3 (13) | **0.0% (<1%)** |
| routing_policy | round-robin* 56.3 (13), least-loaded 57.8 (13), weighted 65.3 (15) | 13.8% |

**Same check at the other width-8 topologies** (interactions):

| base | base tps/GPU (C*) | EP | cache_dtype | max_num_seqs | max_num_batched_tokens | gpu_mem_util | routing_policy |
|---|---|---|---|---|---|---|---|
| tp4·dp2 EP | 89.0 (29) | 6.0% | 0.1% | 0.0% | 0.9% (2048 → 89.8) | 0.0% | 6.3% (weighted 94.9) |
| tp4·dp2 no-EP | 83.6 (30) | 6.0% | 0.1% | 0.0% | **4.6%** (2048/4096 → 79.8) | 0.0% | 3.9% |
| tp2·dp4 EP | 83.0 (32) | 10.5% | 0.0% | 0.0% | 0.0% | 0.0% | 4.0% (weighted *worse*, 80.4) |
| tp1·dp8 EP | 37.9 (14) | 0.0% | 0.0% | 0.0% | 0.0% | 0.7 fails | **37.7%** (least-loaded 60.8, C* 30) |
| tp1·dp8 no-EP | 37.9 (14) | 0.0% | 0.0% | 0.0% | 0.0% | 0.7 fails | 37.7% |

Headline:
- **Topology dominates** (37.9 → 89.0 tps/GPU across the width-8 shapes), then **routing_policy**
  (4–38%, and its best value flips with topology: weighted at tp8/tp4, least-loaded at tp1·dp8,
  round-robin ≈ best at tp2·dp4), then **EP** (0–10.5%).
- **Below 1% everywhere probed: max_num_seqs (0.0%), cache_dtype (≤ 0.5%), gpu_memory_utilization
  (0.0% where it runs).** C* stays ≤ 32, far below every max_num_seqs value, so the batch cap never
  binds and KV memory never runs short.
- **Interaction-only axes**: max_num_batched_tokens is flat at the default topology but moves 4.6% at
  tp4·dp2 without EP (and 0.9% with EP); gpu_memory_utilization is flat in score but 0.7 is infeasible
  at tp1·dp8. Both were kept for that reason.

## Trim to ≤ 5k

11,400 feasible → 4,560 by cutting **max_num_seqs to {32, 256}** (keeping its default and the smallest
value, the only one that could ever bind; it moved nothing at any of the six bases). Nothing else was
dropped: topology axes are never dropped, max_num_batched_tokens and gpu_memory_utilization carry
interactions (above), routing_policy and EP are the strongest engine/CLI axes, and cache_dtype costs
only ×2 (it is the least sensitive kept axis — the next to cut if the space must shrink).

## Cost of the full grid (measured)

`scripts/grid_cost.py` on 300 random feasible configs, fresh cache, `--workers 9`:

| | |
|---|---|
| configs scored | 133 / 300 (44%; matches 8/19 topologies runnable) |
| unique BLIS runs | 4.66 per config (9.25 per scored config, 1 per BLIS-infeasible one) |
| BLIS wall per run | median 0.169 s, mean 0.172 s |
| elapsed | 27.4 s = 0.091 s per config at 9 workers (~51 runs/s) |

A 60 s `cf-grid` trial (fresh cache, 9 workers) did 1,420 configs / 3,017 runs (~55 runs/s), consistent.

**Estimate for the full S-mid grid: ≈ 21,000 unique runs, ≈ 7 min at `--workers 9`** (well under the
brief's n × 6 × median ÷ workers ≈ 4560 × 6 × 0.17 / 9 ≈ 8.6 min bound).

## Incumbent (upstream heuristic) on S-mid

`IncumbentSearcher(refine_top=3)`, budget 400 (fresh cache): stop "done" at **401 runs, 52
evaluations** (19 topologies, then one-knob variants of the top 3; the budget check runs before each
evaluation, hence 401). A budget of 10,000 gives the identical trace — 401 runs is the heuristic's
natural size here.

**Answer: tp 4, dp 2, EP, routing_policy weighted, everything else default → 94.93 tps/GPU
(score −94.925).** Anytime: −82.96 at 50 runs, −88.95 at 100, −90.65 at 200, −94.93 at 400.

The one-knob refinement cannot combine improvements (e.g. weighted routing and
max_num_batched_tokens 2048 at tp4·dp2 each helped alone), so the grid optimum may beat it.

## S-mid v1 ground truth (controller's full grid, `results/s-mid/grid.jsonl`)

4,560 configs, 1,848 scored, 19,024 unique runs, 388 s at 9 workers (estimate above: 21k runs, 7 min).
**Optimum −94.928** at tp4·dp2 + EP, weighted routing, max_num_seqs 32, gpu_memory_utilization 0.7;
**48 configs within 1%**. The incumbent's −94.925 is a **0.003% gap: it found the optimum.** By the
spec's interaction test, S-mid had to be enriched before benchmarking.

## Enrichment (S-mid v2)

The v1 signal was topology × routing, which a topology sweep plus one-knob refinement finds. v2 adds
block_size, scheduler, preemption_policy and admission_policy (token-bucket excluded), and pins the
axes that were under 1% everywhere: cache_dtype fp8, max_num_seqs 256, gpu_memory_utilization 0.9.
It also cuts max_num_batched_tokens to {1024, 2048, 4096, 8192}, because 16384 and 32768 equal 8192
at every base probed. reverse-priority is dropped because it equals priority-fcfs wherever probed.

Sensitivity of the new axes (`scripts/sensitivity.py` on S-large; tps/GPU spread over the axis):

| base | block_size | scheduler | preemption | admission | max_num_batched_tokens | cudagraph |
|---|---|---|---|---|---|---|
| tp8 EP rr (default) | 0.0 | 0.0 | 0.0 | 0.0 | 2.8 | 6.0 |
| tp8 no-EP rr | 0.0 | 0.0 | 0.0 | 0.0 | 5.3 | 3.5 |
| tp4·dp2 EP rr | 0.0 | 0.0 | 0.0 | 0.0 | 5.1 | 4.9 |
| tp4·dp2 no-EP rr | 0.0 | 0.0 | 0.0 | 0.0 | 8.6 | 3.3 |
| tp2·dp4 EP rr | 0.0 | 0.0 | 0.0 | 0.0 | 1.6 | 5.7 |
| tp1·dp8 EP rr | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.3 |
| tp1·dp8 least-loaded | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 6.4 |
| **tp4·dp2 EP weighted** (v1 optimum) | **3.2** (64 best; 16/128 91.9) | 0.0 | 0.0 | 0.0 | **11.6** | 1.1 |
| **tp4·dp2 EP weighted, mnbt 1024** | **7.8** (128: 91.0 vs 64: 83.9) | **7.8** (priority-fcfs 91.0, sjf 87.5) | 0.0 | 0.0 | — | — |
| tp8 EP rr, mnbt 1024, mns 16 | 0.0 | 0.1 | 0.0 | 0.0 | — | — |

So block_size and scheduler matter **only in combination**: block_size only under weighted routing,
and scheduler only when max_num_batched_tokens is small. Both are interaction-only axes, which the
one-knob incumbent cannot see from the default. **preemption_policy and admission_policy
(tier-shed, gaie-legacy) were under 1% at every base probed**. They are kept at 2 values each as the
plan asks, but they are expected to be flat in the grid (×4 configs of exact ties). cudagraph_mode
moves 1–6% but stays out of S-mid (it is in S-large) to keep the size near 5k.

Size: 8 topologies (min_footprint 8) × 4 mnbt × 4 block_size × 3 scheduler × 2 preemption ×
2 admission × 3 routing = **4,608 configs, all at width 8** (only the tp1·dp8 configs can fail).

## Ground truth (S-mid v2)

`cf-grid spaces/glm53flash-h100-chatbot-16.yaml --cache cache --out results/s-mid2/grid.jsonl
--workers 9`: **4,608 configs, all 4,608 scored** (with the floor, tp1·dp8 at gpu_mem 0.9 runs),
43,128 unique BLIS runs (9.36 per config), **1,018 s** at 9 workers (~42 runs/s).

**Optimum −95.143** (95.14 tps/GPU) at tp4·dp2 + EP, weighted routing, **max_num_batched_tokens 2048,
block_size 128, scheduler priority-fcfs** (preemption/admission free: 4 exact ties). It is a
three-way interaction: each of those three values is worse than the default alone at tp4·dp2+weighted
(mnbt 2048: 88.9; block_size 128: 91.9; scheduler alone: flat). **But the optimum is only 0.23% above
the v1 optimum's config** (tp4·dp2+EP+weighted, defaults: 94.925), which is still in the space.

Distribution (tps/GPU): max 95.1, 1st pct 94.3, 10th 88.9, 25th 83.0, median 69.9, 75th 56.3,
90th 38.6, min 37.9; 171 distinct scores. **Plateau: 48 configs within 1%, 72 within 2%, 292 within
5%.** preemption_policy and admission_policy are exact ties throughout (each distinct score ×4).

Baselines (`scripts/baselines_vs_grid.py`, budget 1000; runs = distinct cache keys, cold-cache
equivalent; output in results/s-mid2/baselines.txt):

| method | best tps/GPU | gap to optimum | runs used | evals | stop | runs to within 1% |
|---|---|---|---|---|---|---|
| incumbent (refine_top 3) | 94.925 | **0.23%** | 420 | 44 | done | 203 |
| coord-ascent seeds 0–4 | 66.337 (all seeds) | **30.3%** | 190–207 | 22–24 | converged | never |

**Result, stated plainly: after enrichment the upstream incumbent heuristic still reaches within 1%
of the S-mid optimum (0.23% gap, 420 runs).** Its topology sweep finds tp4·dp2+EP, its one-knob pass
finds weighted routing, and the three-way engine interaction that makes the true optimum is worth only
0.23%. Per the controller's instruction, I stopped enriching here. Coord-ascent fails badly on every
seed for a structural reason: it starts at tp8·dp1, and with the tp·dp = 8 floor, no single-knob
change reaches another topology (tp 4 alone means width 4). So it is stuck at tp8 (best 66.3 with
weighted routing, block_size 16). The 5 seeds differ only in axis order and reach the same point.

Implication for the benchmark: on S-mid v2, the score range that separates methods lies in topology
× routing (37.9–94.9); engine knobs add < 1% at the top. "Within 1%" will be met by the incumbent at
~200 runs. Regret below 0.25% (finding the 4-config exact optimum) is where BSA/TPE could show a gain.

## S-large value check (`spaces/glm53flash-h100-chatbot-16-large.yaml`)

S-mid plus block_size, max_model_len, cudagraph_mode, scheduler, preemption_policy,
admission_policy, routing_scorers (gated on routing_policy = weighted), with finer lists for
max_num_seqs (16…512), max_num_batched_tokens (1024…32768) and gpu_memory_utilization (+0.85).
**45,619,200 feasible configs** with the tp·dp ≥ 8 floor (108,345,600 without it; ≥ 10⁵ either way;
`tests/test_spaces.py` counts analytically).

Valid values come from the pinned source: `--scheduler` {fcfs, priority-fcfs, sjf, reverse-priority},
`--preemption-policy` {fcfs, priority}, `--admission-policy` {always-admit, token-bucket, reject-all,
tier-shed, gaie-legacy}, `--routing-policy` {round-robin, least-loaded, weighted, always-busiest}
(sim/bundle.go); scorers {prefix-affinity, precise-prefix-cache, no-hit-lru, queue-depth,
kv-utilization, load-balance, active-requests, running-requests, load-aware, vllm-dp, lora-affinity}
(sim/routing_scorers.go); cudagraph_mode {NONE, PIECEWISE, FULL, FULL_DECODE_ONLY,
FULL_AND_PIECEWISE} (blis-schemas v0.2.2 rules/v0_29). reject-all and always-busiest were not
considered (degenerate by design).

One real probe per value from the default config (each also ran c = 8 with no error): **BLIS accepted
every value.** tps/GPU (C*) at the default, 56.3 (13) unless noted:

- block_size 16/32/64/128 (and 256): all 56.3. The kernel's `isDSA` is false for glm-5.3-flash
  (head-dim check), so no multiple-of-64 constraint applies, but the runner defaults it to 64.
  **The S-large default is therefore 64**, so S-mid is exactly S-large's sub-space: a stated
  `block_size: 16` gives 66.3 vs 65.3 at tp8 + weighted (block size does matter in combination).
- max_model_len 8192/16384/32768: flat. scheduler (4 values), preemption_policy (2): flat.
- cudagraph_mode: NONE 59.9, FULL / FULL_DECODE_ONLY / FULL_AND_PIECEWISE 59.2, PIECEWISE* 56.3 (6%).
- max_num_batched_tokens 1024: 54.8 (12); max_num_seqs 16: flat; gpu_memory_utilization 0.85: flat.
- admission_policy: tier-shed, gaie-legacy flat. **token-bucket 72.2 (C* 32) — dropped from the
  space**: it sheds offered requests. BLIS counts `injected_requests` = completed + still queued +
  still running + dropped-unservable + timed-out (sim/metrics.go:166), so requests rejected at
  admission are left out of injected and never reach the completion check (c = 16: 75 of 80 offered
  injected; c = 64: 199 of 256), and the throughput/latency describe only the admitted load. It
  would win by refusing load. **Guard:** `CachedRunner.run` now returns `error = "shed: injected=X <
  offered=Y"` for any such run (derived on return, cache unchanged), so a shedding config fails
  the SLO step like any error.
- routing_scorers (weighted): BLIS default `precise-prefix-cache:2,queue-depth:1,kv-utilization:1`
  66.3 (15), kv-utilization:1 66.3, active-requests:1 65.2, running-requests:1 64.7,
  queue-depth:2,kv-utilization:2,load-balance:1 64.4, load-balance:1 57.8, queue-depth:1 / load-aware:1 /
  prefix-affinity:1,queue-depth:1 53.9 (spread 19%).

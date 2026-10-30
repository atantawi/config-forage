# S-mid v3 screening: glm-5.3-flash on 16 H100, chatbot, BLIS v0.14.0

Stack: inference-sim v0.14.0 @ f2ba6bb (kernel v0.4.0, blis-vllm v0.32.0), blis-catalog 0.3.0,
blis-registry v0.2.0, `engine_version: "0.29.0"`. glm-5.3-flash's catalog `config.json` is the same
as in catalog 0.2.1.

Screening space: `spaces/screen/glm53flash-h100-chatbot-16-v014-screen.yaml`. Command:
`scripts/sensitivity.py SPACE --cache cache --base '{...}'`. Raw output:
`results/screen-v014/sensitivity.jsonl` (the records, valid JSONL) and
`results/screen-v014/sensitivity-transcript.txt` (the full stdout, tables included).
Score = tok/s per GPU at c_use = ⌊0.8·C*⌋, ITL p99 ≤ 20 ms.
Spread = (max − min)/max over one axis's values at a fixed base.

## What changed from v0.10.0 on the same configs

**Topology ranking flipped.** Same config (EP on, 2 instances, chatbot, seed 0), ITL p99 in ms at
concurrency 32:

| topology | v0.10.0 | v0.14.0 |
|---|---|---|
| tp8·dp1 | 106.5 | 35.5 |
| tp4·dp2 | 52.7 | 51.9 |
| tp2·dp4 | 19.3 | 47.1 |
| tp1·dp8 | 23.8 | 70.8 |

At v0.10.0, dp-heavy shapes had the lowest tail ITL, and the v2 optimum was tp2·dp4 (132.5 tok/s/GPU).
At v0.14.0, dp ranks step in lockstep (v0.11.0, #1946 "DP lockstep"), so a dp>1 replica is paced by
its slowest rank, and tp8·dp1 is best: 63.5 tok/s/GPU, C* = 12. Mean ITL moves much less than p99.
Lockstep is a tail effect.

**The routing signal collapsed.** Routing gains, v0.10.0 vs v0.14.0:

- **v0.10.0:** in the v2 grid, `routing_policy` had a median one-axis spread of 31.6% (max 41.6%),
  and least-loaded routing alone was worth ~24% at the optimum.
- **v0.14.0, family choice:** routing goes through the llm-d router, and picking the
  `endpoint_picker_config` family is worth 0–17%.
- **v0.14.0, scorer weights:** almost always 0%.

**Knobs that mattered in v2 and are flat now:**

- `max_num_batched_tokens`: v2 up to 16%.
- `scheduler`: v2 up to 9%.
- `block_size`: v2 up to 10%.

**A knob that was flat in v2 and matters now:** `max_num_seqs`. Below 256 it costs 9–47% at
dp>1 and multi-node bases.

## One-axis spreads (%) by base

Defaults: EP on, internal DP LB, load-only router, max_num_seqs 256, cache fp8, gmu 0.9, mnbt 8192,
block 64. "inflight" = `endpoint_picker_config: inflight-load`, "ext" = external DP LB.

| base | tok/s/GPU | EP | max_num_seqs | EPP family | DP LB | kv weight | active weight | queue / prefix weight |
|---|---|---|---|---|---|---|---|---|
| tp8·dp1 | 63.5 | 15.7 | 1.3 | 1.1 | – | **13.6** (0 bad) | – | 0 / – |
| tp8·dp1 inflight | 64.2 | 16.4 | 0.8 | 1.1 | – | 0 | – | 0 / – |
| tp4·dp2 | 50.2 | 15.1 | 13.7 | 3.3 | – | 0 | – | 0 / – |
| tp4·dp2 inflight | 51.9 | – | – | – | – | 0 | 0.7 | 0 / 0 |
| tp2·dp4 | 39.9 | 9.2 | 0.2 | 0.6 | – | 0 | – | 0 / – |
| tp2·dp4 ext | 36.8 | 9.0 | 0.5 | 9.1 | 7.8 | 0 | – | 0 / – |
| tp2·dp4 ext inflight | 40.5 | 9.3 | 2.5 | 9.1 | 2.0 | 0 | – | 0 / – |
| tp1·dp8 | 24.1 | 0 | 9.3 | **17.4** | – | 0 | – | 0 / – |
| tp1·dp8 inflight | 29.1 | 0 | 0.3 | 17.4 | – | 0 | 1.1 | 0 / 0 |
| tp4·dp4 (1 instance, 2 nodes) | 30.0 | 28.8 | 15.4 | 0 | – | 0 | – | 0 / – |
| tp8·dp2 ext inflight (2 nodes) | 54.4 | **57.6** | **47.4** | 16.1 | 12.7 | 0 | 0 | 0 / 0 |
| tp16 (1 instance, 2 nodes) | 56.2 | 37.8 | 36.0 | 0 | – | 0 | – | 0 / – |

**Under 1% at every probed base** (all values scored the same within 1%):

- `cache_dtype`: 0–0.4%, apart from one 1.0% reading.
- `gpu_memory_utilization`, `max_num_batched_tokens`, `block_size`, `scheduler`,
  `preemption_policy`, `queue_scorer_weight`, `prefix_cache_scorer_weight`: 0%.

S-mid v3 pins all of these: the engine and scheduler knobs to one value each, the queue and
prefix-cache weights by having no knob (each family keeps its own weights). They stay unpinned in
S-large, where `prefix_cache_scorer_weight` applies to the prefix-affinity family only and defaults
to that family's weight of 3, so S-mid's configs are S-large's default configs.

## Multi-node

Allowed shapes, as BLIS v0.14.0 places them: width 16 = two whole nodes over `ib-400g`, i.e. tp16,
tp8·dp2, tp4·dp4, tp2·dp8, tp1·dp16, each as one instance.

- **No multi-node shape beats the best single-node shape here.** The best multi-node result is 56.2
  tok/s/GPU (tp16), against 63.5–64.2 for 2 × tp8.
- **Multi-node is where the interactions are.** At tp8·dp2, EP, `max_num_seqs`, the router family
  and DP LB each move the score by 13–58%.
- `--dp-load-balancing hybrid` is refused for a replica that spans nodes, so v3 offers internal and
  external only.
- Accuracy: the BLIS accuracy dashboard does not score multi-node points. These are unvalidated
  predictions.

## Values accepted

Each S-large v3 value was run once at tp8·dp1, 2 instances, c = 8, and all were accepted:

- `cudagraph_mode`: all five. NONE costs 46%.
- `max_model_len`: 8192, 16384, 32768.
- `cache_dtype`: auto, fp8.
- `gpu_memory_utilization`: 0.7, 0.85.
- `max_num_batched_tokens`: 32768.
- `block_size`: 16, 32, 128.
- `max_num_seqs`: 16.

The router accepts weight 0. The `round-robin` picker is not registered in the embedded router, so
there is no round-robin family.

## Request count (num_requests_base 80 → 1280)

One probe point simulates `max(base, 4c)` requests. At the old base of 80 a run covers only ~30–70
s of simulated time and takes ~0.5 s of wall time. The closed-loop chatbot sessions have barely left
their cold start by then (the first round's TTFT is 2–5× the warm one), so throughput is
under-estimated, and by a different amount per topology.

One run at c = 8–16, as tok/s per GPU (wall time):

| config | 80 req | 320 | 1280 | 5120 |
|---|---|---|---|---|
| 2 × tp8 + EP, c = 16 | 86.2 (0.6 s) | 95.5 (1.0 s) | 97.1 (2.7 s) | 97.5 (9.4 s) |
| 2 × tp2·dp4 + EP, c = 16 | 60.4 (1.1 s) | 64.8 (1.6 s) | 67.1 (4.6 s) | 67.6 (17 s) |

Full probe (score at ⌊0.8·C*⌋) for every S-mid v3 topology plus 6 knob variants, against the base:

- **Ranking vs base 2560 (Spearman):** 0.979 at 80, 0.996 at 320, 0.997 at 640, 1.0 at 1280.
- **Size of the error at 80:** typically 2–10%, in both directions. For example, tp1·dp8 scores 24.1
  at 80 and 20.4 at 2560, and the inflight-load gain at tp4·dp2 is 3% at 80 and 12% at 2560.
- **Effect on the screen:** the earlier one-axis screen (base 80) is right about which knobs are
  flat, but its spreads are approximate.
- **The argmax (2 × tp8 + EP) is the same at every base:** 63.5 at 80, 67.1 at 1280.

**At 5120 the default 120 s BLIS timeout cut off dp ≥ 8 configs.** A timeout is not cached, but it
fails that probe point. So `num_requests_base` is now 1280 in both spaces, and the timeout is 900 s.
That costs about 5× the wall time per run. The budget unit (unique runs) is unchanged.

## Pins rechecked at 1280 requests

The one-axis screen above ran at 80 requests, where scores carry 2–10% error. That is more than the
1% pinning threshold. So the screen was repeated at 1280 requests on three bases:

- 2 × tp8 + EP with inflight-load (the S-mid v3 optimum);
- 2 × tp4·dp2 + EP;
- 1 × tp8·dp2 + EP, external LB, inflight-load (2 nodes).

The command is the screening space with `num_requests_base: 1280` and `scripts/sensitivity.py`.

Largest one-axis spread of each pinned axis over the three bases:

| axis | spread |
|---|---|
| cache_dtype | 0.6% (tp4·dp2) |
| max_num_batched_tokens | 0.4% |
| gpu_memory_utilization, block_size, scheduler, preemption_policy | 0.0% |
| queue_scorer_weight, prefix_cache_scorer_weight | 0.0% |

Every pin holds. At 1280 the spreads of the unpinned axes are:

- `max_num_seqs`: 0.4% at tp8, 0.4% at tp4·dp2, 43% at tp8·dp2.
- Router family: 0.1% at tp8, 12% at tp4·dp2, 3% at tp8·dp2.
- EP: 16–54%.
- DP LB: 2–4%.

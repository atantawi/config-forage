# Research log

## 2026-10-28 — Real BLIS: setup, S-mid, sensitivity, cost, incumbent

**Setup and pins.** `scripts/setup_blis.sh ../blis-stack` (idempotent; verifies the
SHA and tags): inference-sim v0.10.0 @ 0d092d4ef64f605046de1181adf8e2b6ee564725 (kernel v0.1.0),
blis-catalog 0.2.1 @ 28e82d4, blis-registry v0.1.1 @ f7519b1, engine_version 0.29.0. Smoke test
`uv run pytest -m blis` passes (tp8 + EP, 2 instances, c = 8; one probe). One BLIS run ≈ 0.17 s.

**Earlier synthetic result (Tasks 1–9).** On the synthetic 6^4 objective, TPE ≥ uniform BSA at
small budgets (B=60: TPE -5.0, SMAC -4.5, BSA -4.0 median over 10 seeds; at B=150 all three reach
the optimum median); a one-cell 'needle' bonus made BSA lose to random, so BSA needs a graded
landscape — the case for biasers rests on the real BLIS benchmark.

**S-mid** (`spaces/glm53flash-h100-chatbot-16.yaml`): glm-5.3-flash (MoE, 64 KV heads), 16 H100,
chatbot, ITL p99 ≤ 20 ms, h = 0.8, fp8 weights. Brief's axes give 11,400 feasible configs; trimmed
max_num_seqs to {32, 256} → **4,560**. Only the 8 width-8 topologies (tp·dp = 8, EP on/off) fit in
BLIS's memory model; the other 11 rule-feasible topologies fail at c = 1, so 1,848 configs score
(1,920 at width 8, less 72 at tp1·dp8 with gpu_mem 0.7).

**Sensitivity** (details: `docs/experiments/s-mid-grid.md`). Default (tp8, EP, round-robin): 56.3
tps/GPU. Topology dominates (37.9–89.0 across width-8 shapes), then routing_policy (4–38%; its best
value depends on topology), then EP (0–10.5%). **Under 1% everywhere: max_num_seqs, cache_dtype,
gpu_memory_utilization** (C* ≤ 32, so neither the batch cap nor KV memory binds).
max_num_batched_tokens is flat over 2048–32768 at the default but moves 4.6% at tp4·dp2 without EP —
an interaction-only axis, kept (v2's added value 1024 costs 2.8% at the default).

**Cost.** Measured on 300 random configs at 9 workers: 4.66 unique runs/config, 0.091 s/config →
full grid ≈ 21k runs, **≈ 7 min**.

**Incumbent** (upstream heuristic, refine_top 3): finishes at 401 runs / 52 evaluations with
**tp4·dp2 + EP + weighted routing = 94.93 tps/GPU**.

**S-large** (`spaces/glm53flash-h100-chatbot-16-large.yaml`): + block_size, max_model_len,
cudagraph_mode, scheduler, preemption_policy, admission_policy, routing_scorers, finer lists →
**108,345,600 feasible** (45,619,200 after the tp·dp ≥ 8 floor, below). Every value was accepted by BLIS in a real probe; token-bucket admission
was excluded because it sheds load (now also caught by the runner's shed guard). Newly sensitive at the
default: cudagraph_mode (6%), routing_scorers (19%); flat: max_model_len, scheduler,
preemption_policy, block_size alone.

### Ground truth

**S-mid v1** (controller's grid, results/s-mid/grid.jsonl): 4,560 configs, 1,848 scored (the 72
non-scoring width-8 configs are tp1·dp8 at gpu_mem 0.7), 19,024 runs, 388 s. Optimum −94.928
(tp4·dp2+EP, weighted, max_num_seqs 32, gpu_mem 0.7); 48 within 1%. The incumbent's −94.925 is a
0.003% gap, so it found the optimum → enrich.

**Fixes.** `model_entry.min_footprint: 8` (feasibility rejects tp·dp < 8: the BLIS memory floor for
glm-5.3-flash on H100). The runner turns a run with injected_requests < offered into an error "shed:
…" (BLIS leaves admission-rejected requests out of injected). S-large is now 45,619,200 feasible.

**S-mid v2** (enriched: + block_size, scheduler, preemption, admission; flat axes pinned): 4,608
configs, all scored, 43,128 runs, 1,018 s. Optimum **−95.143** at tp4·dp2+EP, weighted, mnbt 2048,
block_size 128, priority-fcfs (a three-way interaction). Plateau: 48 within 1%, 292 within 5%.
preemption_policy and admission_policy are exact ties everywhere.

- **Incumbent: −94.925, gap 0.23%, 420 runs, within 1% after 203 runs. It still reaches within 1%
  after enrichment.** The interaction is real but worth only 0.23%. Stopped enriching, per the
  controller.
- **Coord-ascent (5 seeds): −66.337, gap 30.3%, ~200 runs.** Stuck at tp8: under the width-8 floor,
  no single-knob move changes topology.

Takeaway: on this scenario, topology × routing is nearly all the signal. A method beats the
incumbent here only on exact optimality (< 0.25% regret) or on runs to within 1% (< 203).

## 2026-10-28 — Final review findings

The final review of the build branch found three problems that invalidate the first S-mid v2
benchmark (`results/s-mid2/runs`, `results/s-mid2/analysis`). **That benchmark is superseded and must
be rerun, grid included, after these fixes.**

**C1. The uniform `bsa` arm scored configs outside the grid, giving negative regret.** `is_feasible`
applies upstream's one-node rule (tp·dp ≤ 8), so the grid leaves out wider instances. But the
objective never checked feasibility, and BLIS prices multi-node instances (here via the inter-node
fabric). The no-biaser arm sampled tp4·dp4+EP (one 16-GPU instance over two nodes), scored −100.5 to
−101.7, about 7% better than the grid optimum −95.14, and its median regret came out at −5.35. Fix: the
space stays as defined (one instance per node). `Objective` now returns a failure for a rule-infeasible
config without calling BLIS and charges it one run (a synthetic `infeasible:` key, so the same config
costs 1 once). That mirrors BLIS rejecting it at c = 1, so an arm without the feasibility biaser gets
no free feasibility information. `KnobSpace.config` refuses width > budget (num_instances 0), and
cf-analyze raises if a trace beats the grid optimum. Whether multi-node instances should enter the
space is an open question for the user.

**C2. C\* was censored at the ladder top.** `ladder(16)` stops at c = 32, and a config that passed
every rung got C\* = 32. 844 grid configs never failed, among them the optimum and the whole 1%
plateau, so the top of the ranking compared capped values. Fix: when every rung passes, the probe keeps
doubling c (64, 128, …) to the first failure, capped at 4096, then bisects as before;
`diagnostics["c_star_censored"]` marks a config that passes at 4096. The scores of the top configs
change, so the grid, the optimum and every method's regret need a rerun.

**I1. `bsa-diag` was inert on S-mid v2.** Its rule knobs (gpu_memory_utilization, cache_dtype,
max_num_seqs) are single-valued there, and the max_num_batched_tokens rule needs a TTFT SLO that the
scenario does not set. Its factors stayed all ones, and its traces equalled `bsa-feas`'s seed for seed.
`DiagnosticBiaser.active_rules()` now reports the rules that can act; cf-bench records them per arm in
the manifest and warns when there are none; cf-analyze flags arms with identical traces.
(`docs/experiments/s-mid-grid.md`, "The diagnostic arm is inert on S-mid v2".)

Also fixed: TPE's pruned trials are reported under both cost conventions (a "tpe (pruned = 1 run)"
row); trace points carry each evaluation's decision and score, `cf-analyze --heldout N` rescores the
top configs on seeds 1–3, and the table shows the share of seeds within 0.5% of the optimum.

**Rerun (S-mid v2, after the fixes)** (results/s-mid3; grid 4,608 configs all scored, 0 censored,
3 min at 9 workers; benchmark 8 methods × 10 seeds, B = 1000; the grid had already run every config,
so the benchmark needed no new BLIS runs).

- **The capped probe had the wrong optimum.** With C* free to rise above 32, the optimum is
  −132.50 tok/s/GPU at 2 × tp2·dp4 + EP, least-loaded routing, max_num_batched_tokens 4096,
  block 16, C* = 61. The capped grid said −95.14 at tp4·dp2, which is 39% lower and a different
  topology. All 48 configs within 0.5% have C* = 61; 112 are within 1%.
- **Results** (regret = best − optimum in tok/s/GPU; median over 10 seeds):

  | method | regret B=200 | regret B=1000 | runs to 1% | seeds within 0.5% at 1000 |
  |---|---|---|---|---|
  | bsa-feas | 14.2 | 0 | 322 | 10/10 |
  | smac | 20.8 | 0 | 308 | 8/10 |
  | random (feasible) | 14.2 | 0 | 403 (1 censored) | 7/10 |
  | tpe | 29.2 | 0 | 352 (4 censored) | 6/10 |
  | tpe, pruned = 1 run | 29.2 | 0 | 378 (4 censored) | 6/10 |
  | bsa (no biasers) | 28.6 | 14.9 | 732 (5 censored) | 5/10 |
  | incumbent (deterministic) | 25.5 | 1.11 | 235 | 0/10 |
  | coord-ascent | 66.2 | 66.2 | never | 0/10 |

- **Reading.**
  - The upstream incumbent reaches 1% fastest (235 runs), then stops at 0.84% regret. It never
    finds the optimum, because it changes one knob at a time on the best 3 topologies.
  - BSA with the feasibility biaser is the only method within 0.5% on every seed. SMAC is the
    closest rival.
  - No difference between the guided methods is significant at n = 10 (bsa-feas vs tpe:
    p = 0.125 at B = 1000).
  - Feasibility is the one bias that clearly helps: plain BSA, which pays 1 run per infeasible
    sample, is the worst of the samplers.
  - The diagnostic arm is inert in this space. Its rule knobs are pinned, and no TTFT SLO is set.
- **What this does and does not show.** S-mid v2 is a 4.6k-config space: the full grid costs
  ~48k runs, so at B = 1000 all the guided methods approach the optimum. This is a sanity check,
  not the large-space claim. The claim needs S-large (4.6 × 10⁷ configs), where there is no grid
  and the methods are compared on best-found. It also needs a space where the diagnostic rules
  have live knobs.

## 2026-10-28 — PR #1 review fixes

Four findings from the PR #1 review, one commit each.

- **P1. Enumeration on S-large was pathologically slow.** `enumerate_feasible` checked feasibility
  only at the leaves, so finding the first feasible S-large decision walked tens of millions of
  invalid leaves (it did not finish in 5 s). `is_feasible` depends only on the topology knobs and the
  scenario (now a documented invariant), so the DFS prunes at the depth where the last topology knob
  is decided. The first S-large decision now takes ~25 µs. Output and order are unchanged (tested
  against the leaf filter). The new `feasible_topologies(space)` lists the distinct feasible topology
  prefixes. Every searcher's empty-space check and the incumbent's phase 1 use it, so none of them
  walks full decisions.
- **P2a. Plain BSA ignored applicability.** The `bsa` arm (and random with feasible_only=False) had
  no biaser, so on S-large it labelled `routing_scorers` under non-weighted routing. That gave
  aliased configs and wasted budget. Applicability is structure, not a bias:
  `KnobSpace.applicability_biaser()` now always comes first, internally, in BsaSearcher and
  RandomSearcher. Their `biasers` attribute still lists only the given biasers. As a backstop,
  `Objective` canonicalizes each decision (`KnobSpace.canonical`): a stray label gets its unlabelled
  twin's cache keys.
- **P2b. Resume did not detect a changed space.** `space_hash(space)` is a sha256 of the parsed
  scenario and knobs (comments and whitespace do not count). cf-bench stores it in the manifest and
  refuses to resume into another hash, or into no hash. cf-grid writes `<out>.meta.json` with the
  hash and the BLIS pins, and refuses to resume when that sidecar differs, or is missing while the
  grid has rows. cf-analyze `--grid` warns when the runs' hash and the grid's hash differ, or when
  either is missing.
- **P2c. Held-out means ignored failed seeds.** A failed seed now counts as 0 (zero throughput) in
  the held-out mean. Rows also report `failures` and `mean_ok` (the mean over successes), and a
  config with any failure is never a tie or the winner in heldout.md.

**S-mid v2 results (results/s-mid3) are unaffected:** that space has no `requires` (P2a), P1
changes speed only, and P2b adds protection only. Its manifest and grid predate space hashing,
so resuming into them is now refused (start a fresh `--out`). cf-analyze on them warns that the
hash is missing.

## 2026-10-29 — PR #1 second review fixes

Three findings from the second PR #1 review, one commit each.

- **Resume ignored the pins and the code revision.** cf-bench now refuses to resume when the
  manifest's BLIS pins differ from `runner.PINS` (the cf-grid sidecar already did; now tested). The
  code revision is recorded rather than checked, so resuming stays useful while the code changes:
  each trace gets `extra["provenance"]` = {git_sha, dirty, pins}, the manifest keeps the first
  call's `git_sha` and logs {git_sha, dirty, arms, seeds, time} per call (the cf-grid sidecar logs
  git state per call too), and cf-analyze warns, on stderr and in table.md, when the traces span
  more than one revision or any came from a dirty checkout.
- **cf-analyze went on after a definite space mismatch.** When the runs' and the grid's
  `space_hash` both exist and differ, it now raises before writing any output. A missing hash
  (legacy metadata) still only warns.
- **Wide axes could make P negative.** `CategoricalGenerator` validates 0 ≤ eps with eps·n < 1 on
  every axis, and 0 < alpha, rho ≤ 1. A 100-option axis at eps = 0.02 gave min P = −0.683 after one
  update. `default_eps(space)` = min(0.02, 0.5 / widest axis) is the default in BsaSearcher and
  RandomSearcher, so wide axes work with the defaults.

**No existing result is affected:** the shipped spaces' widest axis has 9 options, so eps stays
0.02 and BSA is bit-identical (tested). The pins and the space hash are unchanged. Traces written
before this change carry no provenance. That is not flagged when it is the only gap.

## 2026-10-29 — State at merge of PR #1, and next steps

**Current results:** S-mid v2 with the corrected probe (results/s-mid3; tables in the 2026-10-28
"Rerun" entry and in `docs/experiments/s-mid-grid.md`).

**Spec amended** to match the implemented rules: an infeasible sample costs 1 run, C* is searched
upward without a ceiling, and a run that sheds requests counts as a failure.

Open, in priority order:
1. **Multi-node instances.** Should the one-node rule be relaxed? This is the user's decision. If
   yes, allow widths that are a multiple of 8 up to the budget, then regrid.
2. **S-large benchmark.** No grid; compare methods on best-found against budget. This is the actual
   large-space claim.
3. **A scenario with live diagnostic rules.** This needs a TTFT SLO, or unpinned memory and batch
   knobs, so the diagnostic arm is tested at all.
4. **Small follow-ups:**
   - `cf-analyze --heldout` doesn't check `--space` against the runs' hash.
   - The timeout flag per config.
   - BsaSearcher validates eps at construction.

## 2026-10-29 — Upgrade to BLIS v0.14.0 (engine 0.29.0), multi-node, S-mid v3

User decisions: upgrade first, to v0.14.0 with `engine_version` 0.29.0. Multi-node shapes are
whatever BLIS supports. Router knobs are an EPP family plus scorer weights. RFC #21 (blis-config-search
discussion) waits until the case for BSA over the incumbent is firm.

**Code.**
- **Routing.** v0.12.0 removed BLIS's routing and admission flags. `config_forage/router.py` turns the
  flat knobs (`endpoint_picker_config`, `<scorer>_weight`) into an EndpointPickerConfig, which run_blis
  writes as the scenario's `router:` block. The cache key covers the translated config.
- **DP load balancing.** `data_parallel_load_balancing` is a pool parallel field.
- **Feasibility.** A width above 8 is allowed when it fills whole nodes and divides the budget. Pins
  and the setup script now point at the v0.14.0 stack.

**Findings** (details: `docs/experiments/s-mid-v3-screen.md`):
- **The topology ranking flipped.** On identical configs, v0.14.0's DP lockstep (#1946) raises the
  dp>1 tail ITL (tp2·dp4 p99 at c = 32: 19.3 → 47.1 ms) and lowers tp8's (106.5 → 35.5 ms). The best
  single-axis base is now 2 × tp8·dp1 + EP at ~64 tok/s/GPU. v2's tp2·dp4 optimum gave 132.5.
- **Routing went from the main signal to a minor one.** In v2, routing_policy spread was 31.6%
  median. Now the EPP family is worth 0–17%, and scorer weights ≈ 0 (kv weight 0 is bad at tp8:
  13.6%).
- **`max_num_seqs` now matters** (9–47% at dp>1 and multi-node). mnbt, block_size, scheduler and
  preemption are flat.
- **Multi-node doesn't win here.** Best is 56.2 (tp16). But it has the strongest knob interactions:
  at tp8·dp2, EP 58%, max_num_seqs 47%, family 16% and DP LB 13%.

**Spaces.**
- **S-mid v3:** 3,840 feasible configs, 18 topologies (8 single-node, 10 width-16).
- **S-large v3:** 7.1 × 10⁸ (707,788,800; first 1.4 × 10⁹, halved when the review fixes gated
  `prefix_cache_scorer_weight` to prefix-affinity only).
- **v2 results** (`results/s-mid3`) are archived as the v0.10.0 baseline and are not comparable.
- **Grid:** S-mid v3 ground truth is being computed (`results/s-mid-v3/grid.jsonl`).

## 2026-10-30 — S-mid v3 ground truth (BLIS v0.14.0, 1280 requests per point)

`results/s-mid-v3/grid.jsonl`: all 3,840 configs score (no failures, none censored at c = 4096).
About 27k unique BLIS runs, ~5 h of wall time at 9 workers, with one pause and resume.

- **Optimum: 67.09 tok/s per GPU** at 2 × tp8·dp1 + EP, `inflight-load` router, max_num_seqs 256.
- **The space is nearly flat at the top.** 105 configs are within 1% of the optimum, all at
  tp8·dp1 + EP. 93 of them are within 0.5%. A config sampled at random is within 1% with
  probability 2.7%.
- **The default config scores 67.02, 0.11% from the optimum.** Coord-ascent starts from the default,
  so it is done after one probe. The incumbent probes every topology first (phase 1), so it is not.
- **Topology carries the signal.** The next best topologies are tp8 without EP (−16.3%), tp16 + EP
  across 2 nodes (−16.7%), tp4·dp2 + EP (−19.4%) and tp8·dp2 + EP (−21.2%). Inside tp8 + EP, the
  router family, max_num_seqs and scorer weights move the score by under 1%.

**Consequence:** S-mid v3 cannot separate the search methods. Any method that starts from, or
samples, tp8 + EP finds a config within 1% of the optimum. This is a sanity scenario, not evidence for BSA.
The case for BSA has to come from a scenario where the default topology is not near-optimal and
the knobs interact: the multi-node-required S-mn (glm-5.3, 32 H100) is next.

### Sanity benchmark (S-mid v3, budget 300, seeds 0–4)

`results/s-mid-v3/runs`, `results/s-mid-v3/analysis`. This check confirms the upgraded harness runs
end to end: all 8 arms ran, every BLIS run came from the grid's cache, and cf-analyze and the
held-out rescoring both completed. It is not a method comparison.

| method | runs to 1% (median) | seeds within 0.5% at B = 200 |
|---|---|---|
| coord-ascent | 9 | 5/5 (starts at the default) |
| bsa-feas | 41 | 4/5 |
| random | 41 | 3/5 |
| tpe | 64 | 5/5 |
| smac | 87 | 5/5 |
| incumbent | 105 | 5/5 |
| bsa (no biaser) | 164 | 3/5 |

- **bsa-diag and bsa-feas give identical traces again:** no diagnostic rule fires (no preemption,
  no TTFT SLO).
- **Held-out rescoring:** 75 of the 80 top-10 entries, rescored on seeds 1–3, are within 0.5% of the
  best (−67.05). The others are lower-ranked entries of the incumbent and random. Every method's
  rank-1 config is a tie, so no search winner is a seed-0 artifact.
- **SMAC warnings:** it emits numpy "mean of empty slice" warnings from its local-search timing. They
  are harmless.

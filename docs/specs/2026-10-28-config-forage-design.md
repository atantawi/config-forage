# config-forage: BSA for serving-configuration search — design

Date: 2026-10-28. Status: draft for review.

## 1. Intent

ai-forage built a working Biased Sampling Algorithm (BSA) engine: a categorical generator P over a
decision space, sampled axis by axis as a product of experts (P × biaser factors), with a smoothed,
score-weighted elite update. config-forage applies that engine to a different problem: **finding
good serving configurations for an LLM inference deployment, using the BLIS simulator as the
evaluator.**

The target is the problem `inference-sim/blis-config-search` addresses (discussions #1, #3, #4,
#21): given a model, a GPU type, a GPU budget, a workload and an SLO, find the deployment config
(topology + vLLM engine knobs + llm-d router knobs) that maximizes SLO-feasible throughput per GPU.

**Why.** The existing approaches are exhaustive grids (#21's sweeper) or a rule-driven coordinate
search (`scripts/run_search.py` + `topologies.py`: enumerate topologies, then refine engine knobs
*one at a time* on the best shapes). Both assume a small space. The real space — topology × engine
× router-plugin parameters — is combinatorial (the #21 format reaches thousands of combos with
a handful of knobs), and one-knob-at-a-time cannot see knob interactions by construction. A guided
sampler that concentrates BLIS runs where the good configs are is the gap.

**Positioning (the claim we test).** In a *large* configuration space, BSA reaches a near-optimal
config with a small fraction of the BLIS runs a grid needs, and — the part that is BSA-specific —
its expert factors (exact feasibility, diagnostic rules, transferred priors) make it more
sample-efficient than generic black-box optimizers (TPE, SMAC, BO) at equal budget.

We must be honest that BSA *without* biasers is in the same family as irace/TPE (per-axis
distributions moved toward elites). The contribution stands or falls on the biasers.

## 2. Goals, non-goals, anti-goals

Goals
1. A runnable config-search driver: BSA over a declared knob space, scored by BLIS, with a
   deterministic on-disk cache of BLIS results.
2. A benchmark with ground truth: one mid-size space (~1–5k configs) fully enumerated once, so
   every method gets an exact **regret-vs-BLIS-runs** curve.
3. A large-space demonstration (≥ 10⁵ configs, grid infeasible): best-found vs budget.
4. Baselines at equal BLIS-run budget: random search, Optuna TPE, SMAC3, and the existing
   coordinate search; GP-BO noted/argued but optional (see §7).
5. A biaser ablation: uniform → +feasibility → +diagnostic soft biasers → +transferred prior.
6. Docs-as-you-go: a research log and per-experiment write-ups, ai-forage style.

Non-goals
- Not replacing BLIS feasibility checks; BLIS stays the arbiter, our feasibility biaser is only a
  superset-safe pre-filter.
- Not a report/HTML layer (blis-config-search owns that); we emit their `runs.jsonl`-style records
  so their report can be pointed at our runs later if useful.
- Not live-cluster tuning; simulation only.
- No LLM in the loop for v1 (an LLM prior biaser is an optional later arm).

Anti-goals (would make this a failure even if it runs)
- Claiming BSA beats grid on spaces small enough to grid. The positioning is large spaces.
- Comparing at equal *configs* instead of equal *BLIS runs* (a config's probe costs several runs).
- Tuning BSA's hyper-parameters on the benchmark it is then scored on, without saying so.
- Forking the BSA engine into a drifting copy.

## 3. Constraints and facts (verified 2026-10-28)

- BLIS pinned stack as in blis-config-search `pins.env`: inference-sim `v0.10.0`
  (SHA `0d092d4`), blis-latency-kernel `v0.1.0`, blis-catalog `0.2.1`, blis-registry `v0.1.1`.
  Build per their `.claude/skills/config-search/SETUP.md`. Not installed locally yet; Go is.
- BLIS is deterministic for a fixed seed → results cache exactly by (config, concurrency, seed,
  workload, num_requests) hash.
- Wall time per BLIS run in their committed reports: median ≈ 0.7 s, max ≈ 10 s (307 runs).
  Cheap but not free; a config probe is several runs, so ~5 s/config; a 5k grid ≈ 7 CPU-hours,
  parallelizable.
- Feasibility rules (from their `topologies.py`, verified against v0.10.0): tp×dp×num_instances
  = gpu_budget; dense models take dp=1 and no EP; EP needs tp×dp > 1; one instance ≤ one node
  (8 GPUs) unless the model does not fit; tp must suit the model's heads; weights + min KV must fit.
- Their engine-knob measurements: knobs mostly "bind downward" (lowering max_num_seqs,
  max_num_batched_tokens, gpu_memory_utilization hurts; raising past default changes little).
  That is useful prior knowledge — a candidate soft biaser — and a warning that some engine axes
  are flat, which a benchmark space must not be dominated by.
- Python ≥ 3.12, uv, numpy, pytest — same as ai-forage.

## 4. Decided

- **Repo:** new sister repo at `~/Projects/github/config-forage` (local git, branch `main`).
  It will be **public** (external GitHub), because BLIS and blis-config-search are public and the
  goal is to offer the result to that community as a suggestion. Name: `config-forage`; the
  GitHub account/org that hosts it is pending.
- **Engine reuse:** ai-forage is private, so a public repo cannot depend on it. The BSA engine
  core (`space.py`, `generator.py`, the structural/oracle biasers; numpy only) is **extracted
  once into the public repo** and becomes the canonical copy; ai-forage later switches to depend
  on it. Until then the two copies must not drift: a short note in each records the source commit
  (ai-forage `c4339d2`). LLM-specific parts (`LLMPriorBiaser`, realizer, evaluators) stay private.
- **Start with importance sampling + uniform P, then add biasing** in the ablation order of §2.5.
- **Scalar objective v1:** SLO-feasible output tok/s per GPU at the config's usable capacity,
  as in #3/#21: probe concurrency up and bisect to C* (max concurrency with p99 ITL ≤ SLO, plus
  TTFT if set), score = tokens_per_sec / G at h·C* (h = 0.8 default, configurable). Configs with
  no feasible concurrency or that BLIS rejects score as failures (None → never elites).
  Lower-is-better in the engine, so pass the negative.
- **Budget unit:** BLIS runs (cache hits free, counted separately as "unique runs").

## 5. Shape of the system (builder may refine)

```
spaces/        knob space definitions (YAML): axes, values, applicability, scenario
blis/          runner (scenario YAML + `blis run`), result parsing, content-addressed cache
probe.py       config → C* bisection → score (+ diagnostics extracted at C*)
biasers.py     FeasibilityBiaser, DiagnosticBiaser, TransferPrior  (implement forage.Biaser)
search/        bsa.py, random.py, tpe.py (optuna), smac.py, coord.py (port of their coordinate search)
bench/         grid enumeration (ground truth), runner for (method × seed × budget), plots
docs/          research-log.md, experiments/*.md, specs/
```

- **BLIS runner:** port, don't reimplement, the invocation logic of their `run_blis.py`
  (scenario-file generation, kernel backend via `coefficients:`, metric parsing). Credit it.
- **Axis order matters for feasibility biasing:** model-dependent topology axes first (tp, dp,
  EP), num_instances last so it is *pinned* by the GPU-budget equation (a factor with a single
  nonzero), then engine axes, then router axes. Conditional axes (speculative tokens, PD pool
  sizes, router plugin params of a chosen base config) use `ApplicabilityBiaser`-style None.
- **Diagnostic biaser (v1 idea):** at each generation, read the binding signal of recent elites
  (KV-starved: preemptions > 0 or high KV util; batch-limited; TTFT-bound; etc.) and up-weight
  the knob values that relieve it, with strength β. Soft only; never zero a value.
- **Transfer prior:** P snapshot from scenario A, tempered (P_A^β), as the biaser on scenario B.

## 6. Benchmark scenarios (proposal; builder picks exact values)

- **S-mid (ground truth):** one MoE model in catalog 0.2.1 (glm-5.3-flash or a deepseek), H100,
  chatbot workload, 16–32 GPUs, ITL p99 ≤ 20 ms. Axes: tp, dp, EP, max_num_seqs (5),
  max_num_batched_tokens (5), gpu_memory_utilization (4), cache_dtype (2), routing (3) →
  aim 1–5k feasible configs. Must contain interactions (check after the grid: the best config
  should not be found by one-knob-at-a-time from the default; if it is, enrich the space).
- **S-large:** same scenario plus router plugin parameters (#21 table: scorer weights,
  prefix-cache params, admitter thresholds) and speculative decoding → ≥ 10⁵ configs.
- **S-transfer:** S-mid on H200 or a different workload, for the transfer-prior arm.

Seeds: ≥ 10 per (method, budget) for the stochastic methods; report median and IQR.

## 7. Baselines and the devil's-advocate answer

| Method | Why included | Expected |
|---|---|---|
| Full grid | ground truth on S-mid | optimal, maximal cost |
| Random | floor | |
| Coordinate search (theirs) | the incumbent | cheap, misses interactions |
| Optuna TPE | strongest common choice for categorical/conditional spaces | close to uniform-BSA |
| SMAC3 (RF surrogate) | algorithm-configuration standard | strong at small budgets |
| GP-BO | the "obvious" answer | weak on high-dim categorical + constraints; run only if cheap to set up (e.g. via Ax), else argue |
| BSA ×4 ablation arms | ours | the gap should come from biasers |

Fairness: give TPE/SMAC the same hard constraints the fairest way they allow (constraint
callbacks / forbidden clauses in SMAC's ConfigSpace); infeasible samples they emit still cost a
budget unit only if BLIS is actually called — report both conventions.

## 7b. Calls made by the designer (from the builder check; user may override)

- **Seeds.** Search and ground truth use one workload seed (fixed per scenario). The top 10 configs of
  every method are then re-scored on 3 held-out seeds. Report regret on both, and treat
  differences within 0.5% (upstream's tie tolerance) as ties.
- **Win criterion.** The headline is the BLIS runs needed to come within 1% of the grid optimum (S-mid),
  as a median over ≥ 10 seeds, with a Wilcoxon test of BSA-with-biasers against uniform-BSA and
  against TPE. A secondary measure is regret at budgets of {50, 100, 200, 500, 1000} BLIS runs.
  "Biasers win" means a significant gap against uniform-BSA *and* not losing to TPE.
- **S-transfer.** Best-found vs budget only, no second full grid, unless S-mid turns out cheap
  enough to repeat.
- **Coordinate baseline.** Both a faithful port of upstream's `run_search.py` logic (labelled
  "incumbent") and a generalized coordinate ascent over our full axis lists using our scorer
  (labelled "coord-ascent").
- **Engine location.** A subpackage inside config-forage (`config_forage/bsa/`), not a separately
  published package; ai-forage can depend on config-forage later.
- **Knob support is verified against the pinned runner.** As of v0.10.0, `run_blis.py`
  rejects speculative decoding, PD disaggregation, and `long_prefill_token_threshold`; routing is
  `--routing-policy` / `--routing-scorers` / `--admission-policy` only.
- **S-large (user decision, option c).** Keep the v0.10.0 pins and build S-large from the
  supported knobs only: finer value lists for the batching/memory knobs, plus block_size,
  scheduler, preemption_policy, max_model_len, cudagraph_mode, routing policy/scorers and
  admission policy. Before calling it a large space, check how much each axis actually moves the
  score, and report the axes that barely matter. Add router-plugin parameters, speculative decoding and PD
  as new axes when a pinned BLIS supports them; the space format must make that a pure addition.
- **Language: Python** (user decision). BLIS stays a
  Go binary called as a subprocess.

## 8. Left to the builder

- Exact knob value lists per scenario, after inspecting the catalog and a few BLIS runs.
- Bisection details for C* (bracket, tolerance, the concurrency ladder) — start from their
  `concurrency_ramp`.
- Batch size per generation, α/ρ/scheme defaults (start from ai-forage's tuned values).
- Parallel executor (process pool, host-adaptive cap as in their `sweep_pool.py`).
- Plot styles; the research-log format follows ai-forage's.

## 9. Open questions to revisit

- Pareto (throughput vs per-user rate) instead of the scalar — later.
- Whether C* probing should itself be shared across configs (warm-start brackets from neighbours).
- Contribution path upstream: a `bsa` brain in blis-config-search behind #3's executor/brain split.

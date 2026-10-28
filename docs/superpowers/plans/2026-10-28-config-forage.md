# config-forage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A BSA-driven serving-config search over BLIS, with random, TPE, SMAC and coordinate baselines and a benchmark harness that gives exact regret against a full grid.

**Architecture:** The package is `config_forage`. A YAML knob space compiles to a BSA `DecisionSpace` plus typed values. A cached BLIS runner wraps the vendored upstream `run_blis.py`. A prober turns a config into a scalar score through a concurrency bisection to C*. Every searcher spends a budget counted in unique BLIS runs and emits a trace. A bench layer enumerates the ground truth, runs method × seed, and computes regret and statistics.

**Tech Stack:** Python ≥ 3.12, uv, numpy, pyyaml, optuna (TPE), smac (SMAC3) + ConfigSpace, scipy (Wilcoxon), matplotlib, pytest. BLIS v0.10.0 (Go binary, run as a subprocess).

**Spec:** `docs/specs/2026-10-28-config-forage-design.md`

## Global Constraints

- Pinned stack: inference-sim `v0.10.0` @ `0d092d4ef64f605046de1181adf8e2b6ee564725`, kernel `v0.1.0`, blis-catalog `0.2.1`, blis-registry `v0.1.1`.
- The BSA engine is extracted from ai-forage commit `c4339d2` into `config_forage/bsa/`, numpy only. No LLM code and no imports from `forage`.
- Budget unit = **unique BLIS runs** (cache hits are free and counted separately). Compare methods only at equal unique runs.
- Engine convention: lower score is better. The objective returns `-(tokens_per_sec / gpus)` at the usable concurrency, or `None` for a failed or infeasible config.
- Defaults: headroom h = 0.8; SLO = ITL p99 ≤ 20 ms (plus TTFT p99 if the scenario sets one); tie tolerance 0.5%; "near-optimal" = within 1% of the grid optimum.
- Seeds: ≥ 10 per (method, budget). The search uses one workload seed per scenario; held-out rescoring uses 3 other workload seeds.
- Knobs are limited to those the pinned `run_blis.py` accepts: parallel `tp, pp, dp, enable_expert_parallel`; engine `quantization, cache_dtype, block_size, max_num_batched_tokens, max_num_seqs, max_model_len, cudagraph_mode, gpu_memory_utilization`; CLI `num_instances, scheduler, routing_policy, routing_scorers, admission_policy, preemption_policy`. Adding a knob later must be a pure addition to the space YAML.
- The public repo is licensed Apache-2.0 (designer's call, matching upstream). The vendored file keeps its upstream header plus an attribution line.

## Review Focus

1. **The probe at the bottom of the ladder.** A config that fails the SLO at concurrency 1 must score `None`, never 0 or a tiny number (`test_probe_fails_at_c1_is_none`, Task 4).
2. **A BLIS run that errors mid-bisection** (rc ≠ 0, timeout, or "overloaded"/"incomplete") counts as an SLO failure at that concurrency, not as a crash of the search (`test_probe_treats_run_error_as_fail`, Task 4).
3. **A budget that a batch overshoots.** Searchers stop launching once `runs_used ≥ budget`, and a parallel batch may overshoot. The trace records the true count, and regret is read at the last trace point ≤ the budget (`test_trace_value_at_budget`, Task 9).
4. **Cache key completeness.** Two runs differing only in seed, workload, num_requests or pin must not share a cache entry (`test_cache_key_includes_run_context`, Task 3).
5. **Every sampled sample infeasible.** If the feasibility biaser leaves no tp/dp choice for the budget, the BSA searcher raises a clear error at start-up instead of looping (`test_bsa_rejects_empty_feasible_space`, Task 5).

---

## File Structure

```
pyproject.toml, LICENSE, README.md
config_forage/
  bsa/space.py, bsa/generator.py, bsa/biaser.py   # extracted engine (Task 1)
  space.py           # KnobSpace: YAML -> DecisionSpace + typed values + applicability (Task 2)
  feasibility.py     # is_feasible(config), FeasibilityBiaser (Task 2)
  blis/run_blis.py   # vendored upstream, unchanged except header (Task 3)
  blis/runner.py     # CachedRunner, run_batch (Task 3)
  probe.py           # Slo, probe(), ProbeResult, Objective (Task 4)
  search/base.py     # Trace, Searcher protocol (Task 5)
  search/bsa.py, search/random.py                         (Task 5)
  search/tpe.py, search/smac.py                           (Task 6)
  search/coord.py    # incumbent + coord-ascent          (Task 7)
  biasers.py         # DiagnosticBiaser, TransferPrior    (Task 8)
  bench/grid.py, bench/run.py, bench/analyze.py           (Task 9)
spaces/glm53flash-h100-chatbot-16.yaml  (S-mid), ...-large.yaml (Task 10)
scripts/setup_blis.sh                                     (Task 10)
tests/ ...
docs/research-log.md, docs/experiments/                    (Task 10)
```

---

### Task 1: Scaffold the repo and extract the BSA engine

**Files:**
- Create: `pyproject.toml`, `LICENSE` (Apache-2.0), `README.md`, `config_forage/__init__.py`, `config_forage/bsa/{__init__,space,generator,biaser}.py`
- Test: `tests/test_bsa_space.py`, `tests/test_bsa_generator.py`

**Interfaces:**
- Produces: `config_forage.bsa.space.{Axis, DecisionSpace, Decision}`, `config_forage.bsa.generator.{CategoricalGenerator, InfeasibleSample}`, `config_forage.bsa.biaser.{Biaser, IdentityBiaser, OracleBiaser, ApplicabilityBiaser}`, with signatures identical to ai-forage `c4339d2`.

- [ ] **Step 1:** Run `uv init --package`. Set `name = "config-forage"`, `requires-python = ">=3.12"`, deps `numpy, pyyaml, optuna, smac, scipy, matplotlib`, and dev `pytest`. Add a hatch wheel `packages = ["config_forage"]`.
- [ ] **Step 2:** Copy `forage/space.py` (minus `BINPACK*` and `SPACES`), `forage/generator.py` (verbatim) and `forage/biaser.py` (minus `LLMPriorBiaser`) from `~/Projects/github/ai-forage` at `c4339d2`. Rewrite the imports to `config_forage.bsa.*`. Each file's module docstring gets the line `Extracted from ai-forage c4339d2; this copy is canonical.`
- [ ] **Step 3:** Port `ai-forage/tests/test_generator.py` and `tests/test_space.py`, replacing `BINPACK_SPACE` with a local fixture `SPACE6` that has the same sizes `(4, 4, 3, 3, 2, 3)` so the expected numbers are unchanged. Drop any test that needs `LLMPriorBiaser`.
- [ ] **Step 4:** Run `uv run pytest -q`. Expect all ported tests to pass.
- [ ] **Step 5:** Commit: `Extract the BSA engine from ai-forage c4339d2`.

### Task 2: Knob space and feasibility

**Files:**
- Create: `config_forage/space.py`, `config_forage/feasibility.py`
- Test: `tests/test_space.py`, `tests/test_feasibility.py`, fixture `tests/data/tiny-space.yaml`

**Interfaces:**
- Consumes: `DecisionSpace`, `Axis`, `Biaser` (Task 1).
- Produces:
  - `load_space(path: str | Path) -> KnobSpace`
  - `KnobSpace` (frozen dataclass): `scenario: Scenario`, `knobs: tuple[Knob, ...]`, `decision_space: DecisionSpace`, `config(d: Decision) -> dict` (omits unlabelled axes and fixed knobs' defaults), `decision(config: dict) -> Decision`, `n_combinations() -> int`
  - `Knob`: `name: str, values: tuple, requires: tuple[str, tuple] | None` (the knob applies only when knob `requires[0]` takes one of `requires[1]`)
  - `Scenario`: `model, hardware, workload, gpu_budget: int, engine_version: str, slo: Slo, headroom: float = 0.8, seed: int = 0, num_requests_base: int = 80, model_entry: dict` (catalog row: `arch, heads_q, heads_kv`)
  - `is_feasible(config: dict, scenario: Scenario) -> bool`
  - `FeasibilityBiaser(space: KnobSpace)` implementing `Biaser.factor(axis, partial)`
  - `enumerate_feasible(space: KnobSpace) -> Iterator[Decision]`

Space YAML shape (from the spec, extensible by adding entries):

```yaml
scenario: {model: glm-5.3-flash, hardware: h100, workload: chatbot, gpu_budget: 16,
           engine_version: "0.29.0", slo: {itl_p99_ms: 20}, headroom: 0.8, seed: 0}
knobs:                       # order = sampling order; num_instances is derived, not listed
  - {name: tp, values: [1, 2, 4, 8]}
  - {name: dp, values: [1, 2, 4, 8]}
  - {name: enable_expert_parallel, values: [false, true]}
  - {name: max_num_seqs, values: [64, 128, 256, 512, 1024]}
  - {name: num_speculative_tokens, values: [1, 3], requires: [speculative, [true]]}   # example of conditional form
```

Feasibility rules are copied from upstream `topologies.py` (MoE from `model_entry.arch`):
dense ⇒ dp = 1 and EP false; EP ⇒ tp·dp > 1; tp·dp ≤ 8 (one node); `gpu_budget % (tp·dp) == 0`; tp divides `heads_kv` (when known). `num_instances = gpu_budget // (tp·dp)` is added by `config()`. The weights-fit check is left to BLIS: an out-of-memory rejection becomes a `None` score.

- [ ] **Step 1: Write the failing tests.**
  - `test_load_space_roundtrip`: `config(decision(c)) == c` for a config of the tiny space.
  - `test_config_derives_num_instances`: tp=4, dp=2, budget 16 ⇒ `num_instances == 2`.
  - `test_conditional_knob_unlabelled`: the speculative knob off ⇒ the decision has `None` there and the knob is absent from the config.
  - `test_is_feasible_rules`: one assertion per rule above (a dense model with dp=2 is False; EP with tp=dp=1 is False; tp·dp = 16 is False; budget 12 with tp·dp = 8 is False).
  - `test_feasibility_biaser_never_samples_infeasible`: 2000 samples from `CategoricalGenerator` with `[FeasibilityBiaser]` are all `is_feasible`.
  - `test_enumerate_feasible_matches_filter`: equals the brute-force product filtered by `is_feasible`.
- [ ] **Step 2:** Run `uv run pytest tests/test_space.py tests/test_feasibility.py -q`. Expect FAIL (import errors).
- [ ] **Step 3:** Implement. `FeasibilityBiaser.factor` zeros the options of the current axis that violate a rule given the partial decision. Topology axes come first, so dp is pruned by the chosen tp, and EP by tp·dp. Applicability (`requires`) returns `None`, as `ApplicabilityBiaser` does.
- [ ] **Step 4:** Run the same command. Expect PASS.
- [ ] **Step 5:** Commit: `Knob space YAML and exact feasibility biaser`.

### Task 3: Vendored BLIS runner with cache and parallel batch

**Files:**
- Create: `config_forage/blis/__init__.py`, `config_forage/blis/run_blis.py` (vendored), `config_forage/blis/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `Scenario` (Task 2).
- Produces:
  - `BlisEnv` (dataclass): `blis: Path, catalog: Path, registry: Path, scenarios_dir: Path, timeout: int = 120`, plus `BlisEnv.from_env()` reading `BLIS_BIN, BLIS_CATALOG, BLIS_REGISTRY, BLIS_SCENARIOS`
  - `CachedRunner(env: BlisEnv | None, scenario: Scenario, cache_dir: Path, run_fn=None)`, where `run_fn(config: dict, concurrency: int, seed: int) -> dict` defaults to the vendored `run_one`. Tests inject a fake.
  - `CachedRunner.run(config: dict, concurrency: int, seed: int | None = None) -> dict` returns the upstream record shape (`metrics`, `error`, `returncode`, `wall_s`, `deployment`) plus `cache_key`
  - `CachedRunner.unique_runs: int`, `CachedRunner.cache_hits: int`
  - `cache_key(config, concurrency, seed, scenario) -> str`: the sha256 of the canonical JSON of the config, concurrency, seed, workload, `num_requests` (= upstream `requests_for(c, base)` = `max(base, 4c)`), model, hardware, engine_version and the pin tuple
  - `run_batch(fn: Callable[[T], R], items: list[T], workers: int | None = None) -> list[R]`: a process pool whose default cap is `min(os.cpu_count() - 1, 32)`

- [ ] **Step 1:** Copy upstream `.claude/skills/config-search/scripts/run_blis.py` from inference-sim/blis-config-search, unchanged, and prepend `# Vendored from inference-sim/blis-config-search (Apache-2.0) at <commit sha>; do not edit — re-vendor.` Record the sha with `git -C /tmp/bcs-recon/blis-config-search rev-parse HEAD`.
- [ ] **Step 2: Write the failing tests** (no BLIS, using a fake `run_fn` that counts calls):
  - `test_cache_hit_does_not_call_blis`: the same (config, c) twice ⇒ fake called once, `unique_runs == 1`, `cache_hits == 1`.
  - `test_cache_persists_across_instances`: a new `CachedRunner` on the same `cache_dir` ⇒ 0 new calls.
  - `test_cache_key_includes_run_context`: changing the seed, workload or num_requests base changes the key.
  - `test_run_batch_preserves_order`.
- [ ] **Step 3:** Run `uv run pytest tests/test_runner.py -q`. Expect FAIL.
- [ ] **Step 4:** Implement. The cache is one JSON file per key under `cache_dir/ab/abcdef….json`, written atomically (tmp + rename) so parallel workers are safe. The default `run_fn` calls `run_blis.run_one(env.blis, env.catalog, env.registry, env.scenarios_dir, model, hardware, engine_version, workload, requests_for(c), seed, None, {**config, "concurrency": c}, timeout=env.timeout)`.
- [ ] **Step 5:** Run the same command. Expect PASS.
- [ ] **Step 6:** Commit: `Vendored BLIS runner with content-addressed cache`.

### Task 4: Probe and objective

**Files:**
- Create: `config_forage/probe.py`
- Test: `tests/test_probe.py`

**Interfaces:**
- Consumes: `CachedRunner.run` (Task 3), `KnobSpace.config`, `Scenario` (Task 2).
- Produces:
  - `Slo(itl_p99_ms: float = 20.0, ttft_p99_ms: float | None = None)` with `.passes(metrics: dict) -> bool`
  - `ProbeResult(score: float | None, c_star: int | None, c_use: int | None, tps_per_gpu: float | None, metrics_at_use: dict, diagnostics: dict, runs: int)`
  - `probe(runner, config: dict, scenario: Scenario, seed: int | None = None) -> ProbeResult`
  - `Objective(space: KnobSpace, runner: CachedRunner)`, a callable `(d: Decision) -> ProbeResult` that also keeps `.runs_used`: the number of **distinct `cache_key`s this Objective touched**, whether or not they were cache hits. The grid pre-warms a shared cache, so counting cache misses would make every method look free. `CachedRunner.run` reports the key it used through `record["cache_key"]`.
  - `ladder(gpu_budget: int) -> list[int]`, ported from upstream `concurrency_ramp`

Algorithm (not fixed by the signature):
1. Walk `ladder` upward. A point passes if `record["error"] is None` and `slo.passes(metrics)`. Stop at the first failing point.
2. If concurrency 1 fails, return `score=None`.
3. If the ladder top passes, C* = top. Otherwise bisect on integers between the last pass and the first fail until they are adjacent; C* = the last pass.
4. `c_use = max(1, floor(h·C*))`. Run at `c_use` (often a cache hit). Then `tps_per_gpu = tokens_per_sec / deployment.gpus` and `score = -tps_per_gpu`.
5. `diagnostics`, read from the metrics at C*: `preemptions = preemption_count`, `kv_starved = preemption_count > 0`, `ttft_bound = ttft_p99 fails while itl_p99 passes` at C*+1, and `binding = "itl" | "ttft" | "error"` for the first failure.

- [ ] **Step 1: Write the failing tests**, using a fake runner whose ITL p99 = 5 + 0.5·c ms and tokens_per_sec = 100·c:
  - `test_probe_bisects_to_c_star`: ITL ≤ 20 ⇒ c ≤ 30, so `c_star == 30`, `c_use == 24`, and `score == -(2400 / gpus)`.
  - `test_probe_fails_at_c1_is_none`
  - `test_probe_treats_run_error_as_fail`: the fake returns `error="overloaded: still_queued=3"` for c ≥ 20 ⇒ `c_star == 19`.
  - `test_probe_ladder_top_passes`: C* = the ladder top.
  - `test_probe_counts_unique_runs`: a second `probe` of the same config through the same Objective adds 0 to `runs_used`.
  - `test_runs_used_counts_touched_keys_even_on_cache_hits`: pre-warm the cache with another Objective; a fresh Objective probing the same config gets `runs_used` equal to the distinct keys (> 0), while `runner.unique_runs` does not grow.
  - `test_ttft_slo_applied` when `ttft_p99_ms` is set.
- [ ] **Step 2:** Run `uv run pytest tests/test_probe.py -q`. Expect FAIL.
- [ ] **Step 3:** Implement per the algorithm.
- [ ] **Step 4:** Run the same command. Expect PASS.
- [ ] **Step 5:** Commit: `Concurrency probe to C* and the scalar objective`.

### Task 5: Search interface, BSA and random searchers

**Files:**
- Create: `config_forage/search/{__init__,base,bsa,random}.py`
- Test: `tests/test_search_bsa.py`, `tests/test_search_random.py`

**Interfaces:**
- Consumes: `Objective`, `ProbeResult` (Task 4); `KnobSpace`, `FeasibilityBiaser`, `enumerate_feasible` (Task 2); `CategoricalGenerator` (Task 1); `run_batch` (Task 3).
- Produces:
  - `TracePoint(runs: int, best_score: float | None, best_config: dict | None, evaluated: int)` and `Trace(points: list[TracePoint], method: str, seed: int)` with `.value_at(budget: int) -> float | None` (the best score at the last point with `runs ≤ budget`)
  - `Searcher` protocol: `name: str`, `search(objective: Objective, budget: int, seed: int) -> Trace`
  - `BsaSearcher(space, biasers: Sequence[Biaser] = (), batch: int = 16, alpha=0.7, rho=0.3, eps=0.02, scheme="hard", workers=None, name="bsa")`. The feasibility biaser is *not* implied: arms pass it explicitly, so the uniform arm stays truly uniform.
  - `RandomSearcher(space, feasible_only: bool = True)`

BSA loop: sample `batch` decisions (resampling exact duplicates of already-scored decisions up to 10 tries, then accepting), score them with `run_batch`, call `gen.update(decisions, scores)`, append one `TracePoint` per evaluated config in completion order, and stop once `objective.runs_used ≥ budget`. Log `gen.snapshot()` and `gen.entropy()` per generation on `searcher.history`.

- [ ] **Step 1: Write the failing tests** (a fake Objective on a synthetic 4-axis space whose score has a known unique optimum and an interaction between two axes; 1 run per config):
  - `test_bsa_finds_synthetic_optimum`: budget 300, 5 seeds; the optimum is found in ≥ 4 of 5.
  - `test_bsa_beats_random_median`: the median `value_at(150)` over 10 seeds is better for BSA than for Random.
  - `test_trace_monotone`: best_score never gets worse along the trace.
  - `test_bsa_rejects_empty_feasible_space`: a scenario with gpu_budget 3 and tp ∈ {2, 4} ⇒ `ValueError` before any evaluation.
  - `test_random_feasible_only`: every evaluated config is `is_feasible`.
- [ ] **Step 2:** Run `uv run pytest tests/test_search_bsa.py tests/test_search_random.py -q`. Expect FAIL.
- [ ] **Step 3:** Implement. Check for an empty space with `next(enumerate_feasible(space), None)` (lazy).
- [ ] **Step 4:** Run the same command. Expect PASS.
- [ ] **Step 5:** Commit: `Searcher interface; BSA and random searchers`.

### Task 6: TPE and SMAC baselines

**Files:**
- Create: `config_forage/search/tpe.py`, `config_forage/search/smac.py`
- Test: `tests/test_search_baselines.py`

**Interfaces:**
- Consumes: `Searcher`, `Trace` (Task 5); `KnobSpace`, `is_feasible` (Task 2).
- Produces: `TpeSearcher(space, multivariate: bool = True, n_startup: int = 10)` and `SmacSearcher(space, n_init: int = 10)`, both `Searcher`s; plus `to_configspace(space: KnobSpace) -> ConfigSpace.ConfigurationSpace`.

Fairness, per spec §7:
- TPE uses `optuna.samplers.TPESampler(multivariate=True, group=True, seed=seed)`. Conditional knobs are suggested only when their parent applies. An infeasible suggestion is reported as `optuna.TrialPruned` without calling BLIS, so it costs 0 runs. Record the infeasible count separately on the trace as `trace.infeasible`.
- SMAC uses `HyperparameterOptimizationFacade` on categorical hyperparameters, `EqualsCondition`/`InCondition` for applicability, and `ForbiddenAndConjunction`/`ForbiddenEqualsClause` clauses enumerating the infeasible (tp, dp, EP) triples, so it never proposes them. A failed probe is reported as cost `0.0` (the worst possible throughput), with `status=CRASHED`.

- [ ] **Step 1: Write the failing tests** on the synthetic objective from Task 5 (move the fixture to `tests/helpers.py`):
  - `test_tpe_runs_and_respects_budget`: the trace's last `runs ≤ budget + 1`.
  - `test_smac_never_proposes_infeasible`: every config passed to the objective is feasible.
  - `test_to_configspace_conditions`: the conditional knob is inactive when its parent is off.
  - `test_baselines_deterministic_per_seed`: the same seed gives the same trace.
- [ ] **Step 2:** Run `uv run pytest tests/test_search_baselines.py -q`. Expect FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the same command. Expect PASS.
- [ ] **Step 5:** Commit: `TPE and SMAC baselines with matched constraints`.

### Task 7: Coordinate baselines

**Files:**
- Create: `config_forage/search/coord.py`
- Test: `tests/test_search_coord.py`

**Interfaces:**
- Consumes: Task 5 interfaces; `enumerate_feasible` (Task 2).
- Produces: `IncumbentSearcher(space, refine_top: int = 3)` (`name="incumbent"`) and `CoordAscentSearcher(space, start: dict | None = None, max_sweeps: int = 5)` (`name="coord-ascent"`).

- **Incumbent** follows upstream `run_search.py`: probe every feasible topology (tp, dp, EP) with the engine/router knobs at their defaults, take the best `refine_top` topologies, and probe each with one-knob-at-a-time variants. For each non-topology knob it tries **every** value in our space, not upstream's single `KNOB_TIERS` value. Note this deviation in the module docstring. It is deterministic, so `seed` is ignored.
- **Coord-ascent:** start from the defaults (or `start`). Repeatedly sweep the axes in order; on each axis, try every feasible value with the others fixed and move to the best. Stop when a full sweep makes no move or the budget is spent. If `seed` is given, it permutes the axis order.

- [ ] **Step 1: Write the failing tests:**
  - `test_incumbent_misses_interaction`: on the synthetic objective (optimum reachable only by changing two non-topology axes together), the incumbent ends sub-optimal.
  - `test_coord_ascent_reaches_separable_optimum`: on a separable synthetic objective it finds the optimum.
  - `test_coord_stops_at_budget`.
- [ ] **Step 2:** Run `uv run pytest tests/test_search_coord.py -q`. Expect FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the same command. Expect PASS.
- [ ] **Step 5:** Commit: `Incumbent and coordinate-ascent baselines`.

### Task 8: Diagnostic biaser and transfer prior

**Files:**
- Create: `config_forage/biasers.py`
- Test: `tests/test_biasers.py`

**Interfaces:**
- Consumes: `Biaser` protocol (Task 1), `KnobSpace` (Task 2), `ProbeResult.diagnostics` (Task 4).
- Produces:
  - `DiagnosticBiaser(space, rules: Sequence[Rule], beta: float = 1.0)` with `.observe(results: Sequence[tuple[Decision, ProbeResult]])`, called by `BsaSearcher` after each generation through an optional `observe` hook. `BsaSearcher` calls `b.observe(...)` on every biaser that has the method.
  - `Rule(signal: str, knob: str, prefer: Literal["higher", "lower"] | value)`. The default rule set is `DEFAULT_RULES`: `kv_starved → gpu_memory_utilization higher; kv_starved → cache_dtype fp8; kv_starved → max_num_seqs lower; ttft_bound → max_num_batched_tokens higher; itl binding → max_num_seqs lower`.
  - `TransferPrior(snapshot: list[list[float]], beta: float = 0.5)`, with `factor(axis, partial) = snapshot[axis] ** beta`, and `from_history(path)` to load a P snapshot saved by Task 9's runner.

Diagnostic factor: let `f_s` = the fraction of the last generation's top-ρ elites that show signal `s`. For a matching rule on a knob with ordered values, the factor is `exp(beta · f_s · z)`, where `z` runs linearly from −1 to +1 over the value index in the preferred direction; for a value-target rule, `exp(beta · f_s)` on that value and 1 elsewhere. Factors from different rules multiply, are soft (never 0), and start at 1 before the first `observe`.

- [ ] **Step 1: Write the failing tests:**
  - `test_diagnostic_neutral_before_observe`: all factors are 1.
  - `test_kv_starved_upweights_memory`: after observing elites that are all KV-starved, the factor rises monotonically with `gpu_memory_utilization`, and `cache_dtype=fp8` is greater than 1.
  - `test_diagnostic_never_zero`.
  - `test_transfer_prior_tempering`: β = 0 gives all ones; β = 1 equals the snapshot.
  - `test_bsa_calls_observe`: a spy biaser in `BsaSearcher` sees each generation.
- [ ] **Step 2:** Run `uv run pytest tests/test_biasers.py -q`. Expect FAIL.
- [ ] **Step 3:** Implement, and add the `observe` hook to `BsaSearcher` (Task 5 file).
- [ ] **Step 4:** Run `uv run pytest -q`. Expect everything to PASS.
- [ ] **Step 5:** Commit: `Diagnostic and transfer-prior biasers`.

### Task 9: Benchmark harness — grid, runs, analysis

**Files:**
- Create: `config_forage/bench/{__init__,grid,run,analyze}.py`, CLI entry points in `pyproject.toml`: `cf-grid`, `cf-bench`, `cf-analyze`
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `cf-grid SPACE.yaml --cache DIR --out grid.jsonl [--workers N]`: probes every `enumerate_feasible` decision and writes one JSON line per config (`config`, `score`, `c_star`, `tps_per_gpu`, `runs`). Resumable: it skips configs already in `out`.
  - `cf-bench SPACE.yaml --methods bsa,bsa-feas,bsa-diag,bsa-transfer,random,tpe,smac,incumbent,coord-ascent --seeds 0-9 --budget 1000 --cache DIR --out runs/`: one `trace.json` per (method, seed), plus BSA P history. The method-name → searcher map lives in `bench/run.py` as `ARMS: dict[str, Callable[[KnobSpace, dict], Searcher]]`.
  - `analyze.load_traces(dir) -> list[Trace]`
  - `analyze.runs_to_within(trace, optimum: float, tol: float = 0.01) -> int | None`
  - `analyze.regret_table(traces, optimum, budgets=(50, 100, 200, 500, 1000)) -> dict` with medians and IQRs
  - `analyze.wilcoxon(a: list[float], b: list[float]) -> float`, a p-value with censored runs-to-within set to budget + 1
  - `analyze.rescore_heldout(top_configs, runner, seeds=(1, 2, 3)) -> list[dict]`
  - `cf-analyze runs/ --grid grid.jsonl --out docs/experiments/<name>/` writes `regret.png`, `runs_to_1pct.png`, `table.md`

Each (method, seed) run gets a fresh `Objective` over the shared runner, so `runs_used` stays honest after the grid has warmed the cache (Task 4).

- [ ] **Step 1: Write the failing tests:**
  - `test_each_arm_gets_fresh_objective`: two arms in one `cf-bench` call over a warm cache each report `runs_used > 0`.
  - `test_trace_value_at_budget`: a trace overshooting to 1012 runs reports `value_at(1000)` from the last point ≤ 1000.
  - `test_runs_to_within_censored`: never within ⇒ `None`.
  - `test_grid_resumes`: a second `cf-grid` call evaluates 0 new configs.
  - `test_regret_table_shapes`.
- [ ] **Step 2:** Run `uv run pytest tests/test_bench.py -q`. Expect FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest -q`. Expect everything to PASS.
- [ ] **Step 5:** Commit: `Benchmark harness: grid ground truth, method runs, regret analysis`.

### Task 10: Real BLIS — setup, S-mid space, sensitivity check, first ground truth

**Files:**
- Create: `scripts/setup_blis.sh`, `spaces/glm53flash-h100-chatbot-16.yaml`, `spaces/glm53flash-h100-chatbot-16-large.yaml`, `docs/research-log.md`, `docs/experiments/s-mid-grid.md`, `tests/test_blis_smoke.py` (marked `@pytest.mark.blis`, skipped without `BLIS_BIN`)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1:** `scripts/setup_blis.sh DEST` clones and builds exactly as upstream SETUP.md §1–2 (it verifies the SHA `0d092d4…` and fails loudly if it differs). It clones catalog `0.2.1` and registry `v0.1.1`, then prints the four `export BLIS_*` lines. Run it into `~/Projects/github/blis-stack`.
- [ ] **Step 2:** Write `test_blis_smoke`: one run of `{tp: 8, enable_expert_parallel: true, num_instances: 2}` at c = 8 on glm-5.3-flash/h100/chatbot gives `error is None` and `tokens_per_sec > 0`, and a `probe` gives a non-None score. Run `uv run pytest -m blis -q`. Expect PASS.
- [ ] **Step 3:** Write the S-mid space: glm-5.3-flash, h100, chatbot, budget 16, ITL p99 ≤ 20 ms. Axes: tp {1, 2, 4, 8}, dp {1, 2, 4, 8}, EP {false, true}, max_num_seqs {32, 64, 128, 256, 512}, max_num_batched_tokens {2048, 4096, 8192, 16384, 32768}, gpu_memory_utilization {0.7, 0.8, 0.9, 0.95}, cache_dtype {auto, fp8}, routing_policy {round-robin, least-loaded, weighted}. Print `n feasible` and the estimated cost (n × ~6 runs × measured median wall_s ÷ workers). If more than 5k configs are feasible, drop values on the axes that matter least (Step 4). Pull `heads_kv` and `arch` from the catalog.
- [ ] **Step 4: Sensitivity check.** For each axis, from the default config, run one-axis variation and record the spread of `tps_per_gpu`. Write it to `docs/experiments/s-mid-grid.md`, flagging any axis whose spread is below 1% (the spec requires reporting these). Commit.
- [ ] **Step 5:** Run `cf-grid` on S-mid in the background, then check the spec's interaction test: does `incumbent`'s answer equal the grid optimum? Record the optimum, the incumbent gap, and the score distribution in `s-mid-grid.md`. If the incumbent finds the optimum, note it and enrich the space (add `block_size`, `scheduler`, `preemption_policy`, `admission_policy` values) before benchmarking.
- [ ] **Step 6:** Write the S-large YAML: S-mid plus `block_size, max_model_len, cudagraph_mode, scheduler, preemption_policy, admission_policy, routing_scorers` and finer value lists. Assert at least 10⁵ feasible configs.
- [ ] **Step 7:** Start `docs/research-log.md` with a dated entry giving the setup, pins, S-mid size, sensitivity, optimum and incumbent gap. Commit: `Real BLIS setup, S-mid/S-large spaces, ground-truth grid`.

---

The benchmark runs themselves (all arms × 10 seeds on S-mid and S-large, the transfer arm on S-transfer, held-out rescoring) are experiments that use Task 9's CLIs, logged in `docs/research-log.md`. They are not further build tasks.

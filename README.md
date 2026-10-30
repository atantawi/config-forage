# config-forage

BSA (biased sampling) search over LLM serving configurations, scored by the
[BLIS](https://github.com/inference-sim/inference-sim) simulator.

Given a model, a GPU type, a GPU budget, a workload and a latency SLO, config-forage searches the
deployment space (topology `tp`/`dp`/expert parallelism, single- or multi-node, vLLM engine knobs,
the llm-d router's configuration) for the configuration with the highest SLO-feasible throughput per GPU. The search is a
categorical generator with a smoothed elite update, steered by product-of-experts *biasers*: exact
topology feasibility, diagnostic rules read from BLIS output, and a prior transferred from an
earlier search. The repo also ships the baselines (random, Optuna TPE, SMAC3, the upstream
coordinate heuristic, coordinate ascent) and a benchmark harness that measures every method against
a full grid at equal BLIS-run budgets.

It addresses the problem of
[inference-sim/blis-config-search](https://github.com/inference-sim/blis-config-search), and reuses
its BLIS runner and topology rules (see `NOTICE`). Design: `docs/specs/2026-10-28-config-forage-design.md`.

## Install

Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/):

```sh
uv sync
uv run pytest -q          # unit tests; the BLIS-marked tests skip without BLIS
```

## BLIS setup

The pinned stack is inference-sim v0.14.0 (`f2ba6bb`, latency kernel v0.4.0, blis-vllm v0.32.0),
blis-catalog 0.3.0 and blis-registry v0.2.0. Spaces run `engine_version: "0.29.0"`. Go ≥ 1.27.2 is
required to build the simulator. Results from the earlier v0.10.0 stack (S-mid v2, `results/s-mid3`)
are kept as an archive. Their cache keys carry the old pins, so they never mix with new runs.

```sh
scripts/setup_blis.sh ../blis-stack     # clones, verifies the pinned commits, builds; idempotent
```

It ends by printing four exports; set them in your shell:

| variable | meaning |
|---|---|
| `BLIS_BIN` | the `blis` binary built from inference-sim |
| `BLIS_CATALOG` | the blis-catalog checkout (models, hardware) |
| `BLIS_REGISTRY` | the blis-registry checkout (cost models) |
| `BLIS_SCENARIOS` | a scratch directory for generated scenario files |

`uv run pytest -m blis -q` then runs the real-BLIS smoke test.

## Workflow

A space is a YAML file (`spaces/`): the scenario (model, hardware, workload, GPU budget, SLO,
headroom, requests per point) and the knobs with their values. Adding a knob is a pure YAML addition.
`spaces/screen/` holds wide screening spaces for `scripts/sensitivity.py`. Their one-axis spreads
decide which knobs a benchmark space pins.

```sh
SPACE=spaces/glm53flash-h100-chatbot-16.yaml    # S-mid v3; S-mn: spaces/glm53-h100-chatbot-32.yaml

# 1. Ground truth: probe every feasible config once (resumable).
uv run cf-grid $SPACE --cache cache --out results/s-mid-v3/grid.jsonl --workers 9

# 2. Benchmark: each method x seed to a budget of unique BLIS runs (resumable, shared cache).
uv run cf-bench $SPACE --cache cache --out results/s-mid-v3/runs --budget 1000 --seeds 0-9 --jobs 2 \
    --methods bsa,bsa-feas,bsa-diag,random,tpe,smac,incumbent,coord-ascent

# 3. Analysis: regret vs budget, runs to within 1%, Wilcoxon tests (table.md, regret.png, runs_to_1pct.png).
uv run cf-analyze results/s-mid-v3/runs --grid results/s-mid-v3/grid.jsonl --out results/s-mid-v3/analysis
#    optionally rescore each method's top 10 configs on held-out workload seeds 1-3 (needs BLIS):
uv run cf-analyze results/s-mid-v3/runs --grid results/s-mid-v3/grid.jsonl --out results/s-mid-v3/analysis \
    --heldout 10 --space $SPACE --cache cache
```

Without `--grid` (a space too large to enumerate), cf-analyze reports the best score found instead of
regret. `cache/` holds content-addressed BLIS results, so repeated runs are free.

Resuming is guarded by `space_hash` (a hash of the parsed scenario and knobs) and the BLIS pins.
cf-bench keeps both in `manifest.json`, and cf-grid in `<out>.meta.json`. Each refuses to resume into
results of another space or another BLIS stack, so use a fresh `--out` after editing a space or
upgrading BLIS. The code revision is recorded, not checked: every trace carries
`extra["provenance"]` (git sha, dirty flag, pins), and the manifest and sidecar log each call.
cf-analyze warns, in table.md too, when the traces span several revisions or a dirty checkout. It
refuses runs and a grid whose hashes differ, and warns when either hash is missing. Held-out rescoring counts a failed seed as zero throughput,
and it reports the failures and the mean over successes.

A knob with `requires` (e.g. `active_request_scorer_weight` under
`endpoint_picker_config: inflight-load`) is structural: every
sampler leaves it unlabelled where it does not apply, and the objective canonicalizes decisions, so a
stray label never costs extra runs.

## Router knobs

BLIS (from v0.12.0) routes only through the embedded llm-d router, which an EndpointPickerConfig
configures. A space states it with flat knobs (`config_forage/router.py`):

- `endpoint_picker_config` picks a family: `load-only`, `inflight-load` or `prefix-affinity`. These
  are the configs BLIS ships in `examples/llm-d`.
- `<scorer>_weight` knobs override that family's scorer weights, e.g.
  `kv_cache_utilization_scorer_weight` or `active_request_scorer_weight`.

The runner writes the resulting config next to the scenario. The cache key covers that config, not
just the knob values. `routing_policy`, `routing_scorers` and `admission_policy` are refused.

## Objective and budget unit

A config is probed by raising the concurrency c (upstream's ladder, then doubling up to 4096) until
the SLO (p99 ITL, plus p99 TTFT if set) fails, then bisecting to C*, the largest passing
concurrency. Its score is −(tokens/s per GPU) at c = ⌊h·C*⌋ (h = 0.8 by default); lower is better. A
config that fails at c = 1, or that BLIS rejects, scores None (a failure).

**Budget unit: unique BLIS runs.** One run is one (config, concurrency, seed) simulation; a probe costs
several. Each search counts the distinct runs it touched, cache hits included, so a method's count
does not depend on what other methods ran before it. A config that the topology rules reject is not
sent to BLIS but is charged one run, as if BLIS had rejected it at c = 1. Methods are compared only at
equal unique runs, never at equal configs.

## Results

Two scenarios with full grids as ground truth (BLIS v0.14.0, 1,280 requests per point). Both use the
chatbot workload with an ITL p99 ≤ 20 ms SLO:

| | S-mid v3 | S-mn |
|---|---|---|
| scenario | glm-5.3-flash, 16× H100 | glm-5.3, 32× H100 (multi-node required) |
| configs | 3,840 | 960 |
| optimum (tok/s per GPU) | 67.1 at 2 × tp8 + EP | 25.9 at 2 × tp16 + EP |
| configs within 1% | 105, all in that topology | 21, all in that topology |
| gap to next topology | 16% | 24% |
| default's gap to optimum | 0.11% | 0.51% |
| engine knobs that matter | `max_num_seqs`, only at dp > 1 | none |

In both, one topology (the narrowest tp + EP that fits) carries almost all the signal. Router family
and DP load balancing matter only at dp > 1. A 5-seed sanity benchmark at budget 300 shows that every
method reaches within 1% of the optimum, so neither space can separate the search methods. Runs to
reach 1% (S-mid v3 / S-mn):

- coord-ascent, which starts at the default: 9 / 9
- BSA + feasibility: 41 / 30
- incumbent: 105 / 79

S-mid v3's findings about the shape of the space are posted to
[blis-config-search discussion #21](https://github.com/inference-sim/blis-config-search/discussions/21).
S-mn's are to follow there (`docs/next-steps.md`).

**S-mid v2** (BLIS v0.10.0, 80 requests per point; archived) is not comparable. DP lockstep in v0.11.0
flipped its topology ranking, and its short runs were 2–10% off.

The case for BSA has not been made yet. It needs scenarios where several knobs matter and interact:
a TTFT SLO, the multidoc or agentx workloads, kimi-k2.5. It also needs the large space (S-large) and
a comparison with LLM-designed sweeps. See `docs/next-steps.md`.

Details:
- `docs/research-log.md`: dated entries.
- `docs/experiments/s-mid-v3-screen.md` and `docs/experiments/s-mn-screen.md`: screens, ground truth
  and benchmark tables.
- `docs/experiments/s-mid-grid.md`: the archived v0.10.0 study.

## Caveats

- Simulation only. Scores are BLIS predictions at the pinned stack, not measurements on hardware.
- Multi-node instances (tp·dp a multiple of 8 above one node, over an `ib-400g` fabric) are in the
  space. BLIS prices them, but the accuracy dashboard does not score multi-node points against
  hardware, so they are less validated than single-node ones.
- Both ground truths are chatbot with an ITL-only SLO. Their flatness, the near-optimal default and
  the dead diagnostic biaser (no rule fires) are specific to that setting. In S-mn the default
  topology was chosen before the screen, so its closeness to the optimum is not independent evidence.
- The search uses one workload seed. `cf-analyze --heldout` checks the winners on three others.
- **Probe cost:** a config that fails at c = 1 is charged one run, but on dp16/dp32 shapes that run
  takes 100–250 s at 1,280 requests.

## License

Apache-2.0 (`LICENSE`). Third-party code and rules: `NOTICE`.

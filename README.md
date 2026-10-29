# config-forage

BSA (biased sampling) search over LLM serving configurations, scored by the
[BLIS](https://github.com/inference-sim/inference-sim) simulator.

Given a model, a GPU type, a GPU budget, a workload and a latency SLO, config-forage searches the
deployment space (topology `tp`/`dp`/expert parallelism, vLLM engine knobs, router and admission
policies) for the configuration with the highest SLO-feasible throughput per GPU. The search is a
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

The pinned stack is inference-sim v0.10.0 (`0d092d4`, latency kernel v0.1.0), blis-catalog 0.2.1 and
blis-registry v0.1.1. Go is required to build the simulator.

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
headroom) and the knobs with their values. Adding a knob is a pure YAML addition.

```sh
SPACE=spaces/glm53flash-h100-chatbot-16.yaml

# 1. Ground truth: probe every feasible config once (resumable).
uv run cf-grid $SPACE --cache cache --out results/s-mid/grid.jsonl --workers 9

# 2. Benchmark: each method x seed to a budget of unique BLIS runs (resumable, shared cache).
uv run cf-bench $SPACE --cache cache --out results/s-mid/runs --budget 1000 --seeds 0-9 --jobs 2 \
    --methods bsa,bsa-feas,bsa-diag,random,tpe,smac,incumbent,coord-ascent

# 3. Analysis: regret vs budget, runs to within 1%, Wilcoxon tests (table.md, regret.png, runs_to_1pct.png).
uv run cf-analyze results/s-mid/runs --grid results/s-mid/grid.jsonl --out results/s-mid/analysis
#    optionally rescore each method's top 10 configs on held-out workload seeds 1-3 (needs BLIS):
uv run cf-analyze results/s-mid/runs --grid results/s-mid/grid.jsonl --out results/s-mid/analysis \
    --heldout 10 --space $SPACE --cache cache
```

Without `--grid` (a space too large to enumerate), cf-analyze reports the best score found instead of
regret. `cache/` holds content-addressed BLIS results, so repeated runs are free.

Resuming is guarded by `space_hash` (a hash of the parsed scenario and knobs). cf-bench keeps it in
`manifest.json`, and cf-grid keeps it with the BLIS pins in `<out>.meta.json`. Each refuses to
resume into results of another space, so use a fresh `--out` after editing a space. cf-analyze warns
if the runs and the grid hashes differ. Held-out rescoring counts a failed seed as zero throughput,
and it reports the failures and the mean over successes.

A knob with `requires` (e.g. `routing_scorers` under `routing_policy: weighted`) is structural: every
sampler leaves it unlabelled where it does not apply, and the objective canonicalizes decisions, so a
stray label never costs extra runs.

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

See `docs/research-log.md` (dated entries) and `docs/experiments/` for the ground truth, sensitivity
analyses and benchmark results.

## Caveats

- Simulation only. Scores are BLIS predictions at the pinned stack, not measurements on hardware.
- One instance must fit one node (tp·dp ≤ 8 GPUs), as in upstream's topology rules. BLIS will price
  multi-node instances, which can score higher; they are outside the space by definition.
- The S-mid ground truth is one scenario (glm-5.3-flash, 16 H100, chatbot). Its signal is mostly
  topology × routing, the upstream incumbent heuristic gets within 1% there, and the diagnostic
  biaser has no active rule in it (see the research log). Results do not generalize beyond it
  without further scenarios.
- The search uses one workload seed. `cf-analyze --heldout` checks the winners on three others.

## License

Apache-2.0 (`LICENSE`). Third-party code and rules: `NOTICE`.

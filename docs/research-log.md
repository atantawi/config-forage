# Research log

## 2026-10-28 — Real BLIS: setup, S-mid, sensitivity, cost, incumbent

**Setup and pins.** `scripts/setup_blis.sh ~/Projects/github/blis-stack` (idempotent; verifies the
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
max_num_batched_tokens is flat at the default but moves 4.6% at tp4·dp2 without EP — an
interaction-only axis, kept.

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

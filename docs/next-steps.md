# Next steps

The standing list of open work. The research log is dated history; this file is the current plan.
Update it when an item is done or re-prioritized.

## Scenarios that can separate the search methods (decided with the user, 2026-10-30)

S-mid v3 (glm-5.3-flash, 16 H100) and S-mn (glm-5.3, 32 H100) are both **chatbot with an ITL-only
SLO**. In both, one topology dominates, and almost every engine knob is flat (under 1%). The
default topology is at or near the optimum, and one-knob-at-a-time search finds it at once. Neither
scenario can show a BSA advantage. Both keep their value as findings about the search space (posted
to blis-config-search discussion #21).

The next scenarios should make more engine knobs matter:

1. **Add a TTFT SLO** (e.g. `ttft_p99_ms`) to the same scenarios. Prefill knobs
   (`max_num_batched_tokens`, chunking) and the router then become binding. This also gives the
   diagnostic biaser live rules for the first time: `ttft_bound` → `max_num_batched_tokens`.
2. **A different workload:**
   - `multidoc`: long, shared-prefix prompts. Prefix caching, the prefix-affinity router, and KV
     memory should matter.
   - `agentx-mvp`: agentic replay with long contexts. KV pressure brings in `max_num_seqs`,
     `gpu_memory_utilization`, `cache_dtype` and `max_model_len`.
3. **kimi-k2.5 on H100.** Whether one node fits depends on the engine knobs (gmu 0.95, or fp8 KV
   with max_model_len ≤ 16k), so memory and topology interact.

Before choosing, screen each candidate at 1280 requests. Check that more than the topology carries
signal, then grid the pinned space.

## LLM-designed sweeps as a search arm (decided with the user, 2026-10-30)

Add upstream's skill-driven experiment design ([blis-config-search PR #22](https://github.com/inference-sim/blis-config-search/pull/22)) to the benchmark portfolio, next to BSA, the incumbent and the baselines. **Test it on the next scenario** (a TTFT-SLO or multidoc/agentx one), not on S-mid v3 or S-mn. Their findings are public (RFC #21) or about to be, so they would contaminate the test.

- **Arm.** The skill writes a spec for the scenario. We expand the spec to its configs, keep those that match our GPU budget, and score each with our probe. Scoring with the spec's own concurrency list is a second variant. The trace follows the spec's order; the budget is unique BLIS runs. LLM tokens and wall time go in a separate cost column.
- **Clean room.** Run the skill as a fresh agent in a worktree with no `results/`, `docs/`, research log or grid data, and no access to this repo's findings. Note that the skill's own reference notes contain measured findings, which counts as legitimate domain knowledge.
- **Variation.** The skill is nondeterministic, so run it 5–10 times and treat each run as a seed.
- **Budget.** A spec has no notion of a run budget. Either ask the skill to fit one, or compare at the size it chooses. Record which was done.
- **Pins.** Pin a specific PR #22 commit. PR #22 targets BLIS v0.12.0 and we run v0.14.0; the knob names match RFC #21, so the translation is mechanical.
- **Hybrid.** Use an LLM-designed spec as BSA's prior: the LLM proposes, BSA refines. Compare it against both on their own.

## Other open items

- **S-large benchmark:** no grid; compare the methods on best-found against budget.
- **Second set for RFC #21:** post S-mn's findings in the same thread, linking `main` once PR #3 is merged. The draft has been shown to the user.
- **Feedback on blis-config-search PR #22:** try the experiment skill and leave review comments on the PR. Raise the `max(80, 4c)` requests per run at `scripts/run_experiment.py:603`.
- **Cheap c = 1 pre-check:** a config whose ITL at c = 1 already breaks the SLO fails after one run,
  but at 1280 requests that run takes 100–250 s on dp16/dp32 shapes. A short run at c = 1 (e.g. 80
  requests) could reject it first. It must not change any scored result: a config that fails short
  but might pass long still gets the full run.
- **Small follow-ups:**
  - `cf-analyze --heldout` doesn't check `--space` against the runs' hash.
  - A per-config timeout flag.
  - BsaSearcher validates eps at construction.

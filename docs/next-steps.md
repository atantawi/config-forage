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

## Other open items

- **S-large benchmark:** no grid; compare the methods on best-found against budget.
- **Second set for RFC #21:** post S-mn's findings in the same thread. Show the user the draft first.
- **Small follow-ups:**
  - `cf-analyze --heldout` doesn't check `--space` against the runs' hash.
  - A per-config timeout flag.
  - BsaSearcher validates eps at construction.

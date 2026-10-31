# Next steps

The standing list of open work. The research log is dated history; this file is the current plan.
Update it when an item is done or re-prioritized.

## Scenarios that can separate the search methods (decided with the user, 2026-10-30)

S-mid v3 (glm-5.3-flash, 16 H100) and S-mn (glm-5.3, 32 H100) are both **512/512 `distribution` with an
ITL-only SLO** (labelled chatbot until 2026-10-30: closed-loop BLIS ignored `--workload`). In both, one topology dominates, and almost every engine knob is flat (under 1%). The
default topology is at or near the optimum, and one-knob-at-a-time search finds it at once. Neither
scenario can show a BSA advantage. Both keep their value as findings about the search space: S-mid
v3's and S-mn's are both posted to blis-config-search discussion #21 (2026-10-30).

**RFC #21 corrected (2026-10-31).** A [correction](https://github.com/inference-sim/blis-config-search/discussions/21#discussioncomment-19314444)
is posted, and both original posts carry a note pointing to it. blis-config-search had already found
and fixed the bug in its own runner (PR #19, 2026-10-29); ours was vendored before that. No
inference-sim issue has been filed (the user's choice); BLIS v0.14.0 still ignores `--workload` with
`--concurrency`.

**Screened (2026-10-31): real chatbot on glm-5.3-flash, 16 H100** (`docs/experiments/chatbot-screen.md`).
It has an engine knob whose effect depends on topology, and an optimum away from the default, which
S-mid v3 and S-mn lacked:

- **S-chat** (`spaces/glm53flash-h100-chatbot-real-16.yaml`, 7,680 configs): chatbot, ITL p99 ≤ 20 ms.
  Topology ranking reversed from S-mid v3 (1 × tp8·dp2 + EP 96.2 vs 2 × tp8 + EP 42.0 tok/s/GPU), and
  `max_num_seqs` is worth 74–84% at the one-instance shapes but under 1% at the others. **The main BSA
  benchmark.** Grid running (2026-10-31).
- **A second real-workload scenario** was screened alongside it and is reserved as **the clean-room
  scenario for the LLM-designed-sweep arm** (decided with the user, 2026-10-31). This repo is public, so
  its space, screen and grid stay out of the repo until that test has run.
- Dropped: summarization. Still open: agentx-mvp (needs replay support) and kimi-k2.5 on H100.

## LLM-designed sweeps as a search arm (decided with the user, 2026-10-30)

Add upstream's skill-driven experiment design ([blis-config-search PR #22](https://github.com/inference-sim/blis-config-search/pull/22)) to the benchmark portfolio, next to BSA, the incumbent and the baselines. **Test it on the reserved clean-room scenario**, not on S-mid v3, S-mn or S-chat. Their findings are public (RFC #21) or about to be, so they would contaminate the test.

- **Arm.** The skill writes a spec for the scenario. We expand the spec to its configs, keep those that match our GPU budget, and score each with our probe. Scoring with the spec's own concurrency list is a second variant. The trace follows the spec's order; the budget is unique BLIS runs. LLM tokens and wall time go in a separate cost column.
- **Clean room.** Run the skill as a fresh agent in a worktree with no `results/`, `docs/`, research log or grid data, and no access to this repo's findings. Note that the skill's own reference notes contain measured findings, which counts as legitimate domain knowledge.
- **Variation.** The skill is nondeterministic, so run it 5–10 times and treat each run as a seed.
- **Budget.** A spec has no notion of a run budget. Either ask the skill to fit one, or compare at the size it chooses. Record which was done.
- **Pins.** PR #22 was merged on 2026-10-30; pin the blis-config-search `main` commit the arm uses. It targets BLIS v0.12.0 and we run v0.14.0; the knob names match RFC #21, so the translation is mechanical.
- **Request count.** The merged sweeper still runs `max(80, 4c)` requests (`scripts/run_experiment.py:613`). The arm scores configs with our 1280-request probe, so this does not affect it. The issue is already raised in our first RFC #21 comment.
- **Hybrid.** Use an LLM-designed spec as BSA's prior: the LLM proposes, BSA refines. Compare it against both on their own.

## Other open items

- **S-large benchmark:** no grid; compare the methods on best-found against budget.
- **Cheap c = 1 pre-check:** a config whose ITL at c = 1 already breaks the SLO fails after one run,
  but at 1280 requests that run takes 100–250 s on dp16/dp32 shapes. A short run at c = 1 (e.g. 80
  requests) could reject it first. It must not change any scored result: a config that fails short
  but might pass long still gets the full run.
- **Small follow-ups:**
  - `cf-analyze --heldout` doesn't check `--space` against the runs' hash.
  - A per-config timeout flag.
  - BsaSearcher validates eps at construction.

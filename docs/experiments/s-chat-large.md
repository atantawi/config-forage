# S-chat-large: every knob free on the real chatbot preset (2026-11-01)

Space: `spaces/glm53flash-h100-chatbot-real-16-large.yaml`, S-large v3's knobs on chatbot, ITL p99 ≤ 20 ms,
9,953,280,000 combinations. S-chat is a sub-space of it with the same defaults (checked by
`tests/test_spaces.py`). There is no grid, so methods are compared on the best score found against
budget.

## Benchmark (budget 1000, 10 seeds)

`uv run cf-bench spaces/glm53flash-h100-chatbot-real-16-large.yaml --cache cache --out results/s-chat-large/runs --budget 1000 --seeds 0-9 --jobs 15 --workers 1 --methods bsa,bsa-feas,random,tpe,smac,incumbent,coord-ascent`:
54,910 new BLIS runs, 1 h 17 min on the M5 Max. Best found by any method: **108.9 tok/s per GPU**.

| method | final best, median [IQR] | seeds within 1% of 108.9 |
|---|---|---|
| tpe | 106 [92.7, 109] | **5/10** |
| incumbent (deterministic) | 105.8 | 0/10 |
| random | 106 [99.1, 106] | 0/10 |
| smac | 103 [88.3, 106] | 1/10 |
| bsa-feas | 97.7 [88.3, 104] | 1/10 |
| coord-ascent | 97.1 [88.3, 106] | 1/10 |
| bsa | 96.1 [88.2, 106] | 3/10 |

- TPE again leads, though not significantly at this size (TPE vs BSA at B = 1000: p = 0.21). BSA has
  the widest spread: 3 seeds reach 108.9, but its worst seed ends at 34.2.
- The best config is 1 × tp8·dp2 + EP, S-chat's optimum topology, with `cudagraph_mode: FULL`,
  external DP LB, the inflight-load router, `max_num_batched_tokens` 32768 and `max_num_seqs` 512.
- The traces were written from a checkout whose only uncommitted change was this space and its test;
  the code was `main` at d90bbb3. They predate the boltzmann NaN fix (PR #10), which does not touch
  the hard scheme used here.

## `cudagraph_mode` on chatbot

The benchmark's best is 13% above S-chat's grid optimum (96.2), because S-chat pins
`cudagraph_mode` to PIECEWISE, the runner's default. The S-mid v3 screen had not screened it (its
"values accepted" check only ran each value once). A screen with `cudagraph_mode` and
`max_model_len` added (`spaces/screen/glm53flash-h100-chatbot-16-real-screen.yaml`, 9 bases, 323
probes, 3 min):

| base | NONE | PIECEWISE | FULL | FULL_DECODE_ONLY | FULL_AND_PIECEWISE |
|---|---|---|---|---|---|
| 1 × tp8·dp2 + EP | 8.6 | 96.2 | **105.8** | 16.7 | 99.2 |
| same, inflight router + external LB | 8.5 | 96.1 | **108.9** | 16.3 | 99.2 |
| 1 × tp16 + EP | 9.9 | 77.6 | **88.3** | 21.9 | 80.8 |
| 2 × tp8 + EP | 22.4 | 42.0 | 44.4 | 44.1 | **44.5** |
| 2 × tp4·dp2 + EP | 15.4 | 26.4 | **27.6** | 27.4 | 27.5 |
| 2 × tp2·dp4 + EP | 9.9 | 20.4 | **21.1** | 20.9 | **21.1** |

- FULL is best or within 0.3% of best at every base, worth +10–14% at the one-instance shapes.
- FULL_DECODE_ONLY is a second topology-dependent trap, like low `max_num_seqs`: near-best at the
  two-instance shapes, −72% to −83% at the one-instance ones (C* falls to 3).
- NONE costs 47–92% everywhere. `max_model_len` is flat (0.0%) at every base.
- The other knobs' spreads are as in the earlier screen, with FULL as well as PIECEWISE.

**Caveat on S-chat (decided with the user, 2026-11-01).** S-chat's grid, benchmark and BSA tuning ran
with `cudagraph_mode` pinned to PIECEWISE, so 96.2 is the optimum of that pinned space, not of the
knob space; with `cudagraph_mode` free it is at least 108.9. S-chat is not regridded. The pin is not
expected to change the comparison of methods: S-large, with the knob free, gives the same picture,
and the FULL_DECODE_ONLY trap is of the same kind as the one S-chat already has.

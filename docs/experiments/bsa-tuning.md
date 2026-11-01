# BSA generator tuning on S-chat (2026-11-01)

**Question.** On S-chat, BSA's sampling distribution concentrates on 1 × tp16 + EP, 19% below the
optimum, within about 300 runs. On those seeds it never reaches the optimum. Can the generator's
settings fix that?

**Method.** `scripts/bsa_sweep.py` runs BsaSearcher on a space whose grid is cached (every probe is a
cache hit, so no BLIS runs) and reports runs to within 1% of the grid optimum. The sweep covered
48 settings on seeds 0–19, budget 1000:
- smoothing α ∈ {0.7, 0.3, 0.1}, batch ∈ {16, 32, 64} and elite fraction ρ ∈ {0.3, 0.1};
- the rank and boltzmann elite schemes;
- ε = 0.05;
- each with and without the feasibility biaser.

Five settings were then rerun on **held-out seeds 20–39**, next to TPE, random and coordinate ascent
on the same seeds. They were picked so as not to fit S-chat alone; the selection also used the
reserved scenario, whose results are not published here.

**Held-out results on S-chat** (runs to within 1%, median; censored = never within 1%, of 20):

| method | median | censored |
|---|---|---|
| **tpe** | **240** | 2 |
| BSA α 0.3, batch 16, ρ 0.1 | 276 | 3 |
| BSA default (α 0.7, batch 16, ρ 0.3) | 348 | 4 |
| BSA α 0.3, batch 16, ρ 0.1 + feasibility | 354 | 0 |
| BSA rank scheme + feasibility | 462 | 5 |
| BSA default + feasibility | 518 | 3 |
| random | 556 | 3 |
| coord-ascent | 1001 | 11 |

- **No setting beats TPE.** The best setting on seeds 0–19 (α 0.3, ρ 0.1, feasibility: median 226)
  falls to 354 on fresh seeds. Most of its lead was selection on noisy medians.
- **Slower smoothing and a smaller elite set help reliability more than speed.** With feasibility,
  α 0.3 / ρ 0.1 reached 1% on all 20 held-out seeds, the only BSA setting to do so, but at a
  median of 354.
- The default BSA's median moves between 348 and 573 across seed sets, so single 10-seed medians are
  not a reliable basis for comparing BSA settings.

**Bug found and fixed.** Under the boltzmann scheme, an elite group can contain samples far enough
behind the best that their weight underflows to exactly 0. When the only elites labelling a gated
axis all weighed 0, its update divided 0 by 0 and the axis became NaN. That crashed S-chat seed 4.
The axis is now left unchanged, as with no labelled elites. Regression test:
`tests/test_bsa_generator.py::test_axis_whose_labelled_elites_all_weigh_zero_is_unchanged`. The fix
does not affect the hard scheme, which every benchmark so far used.

The sweep records are local (`results/s-chat/bsa-sweep*.jsonl`).

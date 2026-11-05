# canonical_config folds and inert eplb fields, re-measured on BLIS v0.15.0

Stack v0.15.0: inference-sim v0.15.0 @ 762f160 (kernel v0.5.0, blis-vllm v0.32.0-blis.1, schemas
v0.10.0, router v1.0.0-blis.2), blis-catalog 0.5.0, blis-registry v0.3.0, `engine_version: "0.32.0"`.
Script: `scripts/measure_folds.py` (output below verbatim). Each claim that v0.14.0 measured
("measured identical") was measured again here before it was let onto v0.15.0
(`Stack.measured_folds`, `Stack.inert_block_fields` in `config_forage/blis/stacks.py`).

**Outcome.** On v0.15.0 the folds `long_prefill_token_threshold_zero`, `all2all_default`,
`eplb_disabled` and `eplb_zero_redundant` apply, and eplb `window_size` and `step_interval` are
refused as inert. `all2all_alias` does **not** apply: pplx and naive are priced exactly as
allgather_reducescatter (every metric is equal), but the Result document also records the engine's
rename warning in `kernel.colocated.skipped_rule_findings` (`all2all-backend-known`: "engine 0.32.0
accepts "pplx" but runs "allgather_reducescatter" in its place"). So the runs are not byte-identical,
and on v0.15.0 an alias keeps its own cache key and its own run.

## What is compared

BLIS v0.15.0 prints one JSON `kind: Result` document on stdout. The script does not compare the flat
metrics a record keeps (`result_doc.parse_result_document`): a fold must leave every field the same,
not only the ones scoring reads. It compares the whole document, parsed as JSON, after dropping
only these fields:

| dropped field | why it differs between two equivalent runs |
|---|---|
| `name` | the run's slug, built from the scenario file name (a hash of the deployment as written) |
| `input.scenario.name`, `input.deployment.name` | the scenario file name, as above |
| `input.deployment.pools[*].engine` | the echo of the engine block as written: a variant states the value its canonical form leaves out |
| `runtime.wall_clock_s`, `runtime.started_utc` | wall-clock timing of the run |

Two runs of one config differ only in `runtime`. Everything else, including `metrics`, `requests`,
`engine`, `kernel` (assumptions and rule findings) and `breakdowns`, must be equal. On v0.14.0, whose
stdout is not a Result document, the script compares every JSON object on stdout.

Each comparison runs one instance of tp4·dp2 with EP at c=64, the shape the v0.14.0 claims were
measured on, with quantization fp8 and cache_dtype fp8 as the spaces pin them. Folds and refusals are
turned off for the measurement, so every variant reaches BLIS as written. Two controls (values BLIS
prices) must differ, or the run is refused as proving nothing.

## glm-5.3-flash does not run on engine 0.32.0

The v0.14.0 claims were measured on glm-5.3-flash/h100/chatbot. On v0.15.0 with engine 0.32.0 that
model fails before any step:

```
scenario "glm-5.3-flash-h100-….yaml": layer kind kda_dense: recurrent state is not mapped to a KV cache spec
```

The message comes from blis-vllm v0.32.0-blis.1 `compat/kernelmodel/kvlayers.go:150` (`kvLayersOf`).
The 0.32.0 engine does not build hybrid KV groups yet; `V032Pieces` in
`compat/kernelmodel/kernelmodel.go:516-531` lists it. qwen3.5-397b-a17b-fp8 and qwen3.6-35b-a3b fail
the same way (`layer kind gdn_moe: …`). On the same v0.15.0 binary, engines 0.29.0 and 0.31.0 do run
glm-5.3-flash and qwen3.5-397b-a17b-fp8 at this shape (rc 0). They go through the compatibility engine
(inference-sim `cmd/engine_driver_build.go:23,34`). No workaround is added here.

So v0.15.0 is measured on two other MoE models that run on engine 0.32.0, and a claim goes into the
v0.15.0 tables only if it holds on both:

- **kimi-k2.5/h200/chatbot** (384 experts). At tp4·dp2 the default gmu 0.9 leaves no KV room with
  8 redundant experts or more (126.9 GiB of fixed occupancy against a 125.8 GiB budget). So the base
  sets gmu 0.95, and the EPLB variants use 8 and 16 redundant experts (392 and 400 both divide by 8).
- **minimax-m2.7/h100/chatbot** (256 experts), with 32 and 64 redundant experts as on v0.14.0.

## kimi-k2.5/h200/chatbot, v0.15.0

```
uv run python scripts/measure_folds.py v0.15.0 --model kimi-k2.5 --hardware h200 --redundant 8,16 \
    --set '{"gpu_memory_utilization": 0.95}'
```

stack v0.15.0, engine 0.32.0, kimi-k2.5/h200/chatbot, 1 x tp4*dp2 + EP, c=64, 1280 requests, seed 0; base {'tp': 4, 'dp': 2, 'enable_expert_parallel': True, 'num_instances': 1, 'quantization': 'fp8', 'cache_dtype': 'fp8', 'concurrency': 64, 'gpu_memory_utilization': 0.95}

compared: BLIS stdout less name, runtime.wall_clock_s, runtime.started_utc, input.scenario.name, input.deployment.name, input.deployment.pools[*].engine

| fold / field | claim | identical |
|---|---|---|
| long_prefill_token_threshold_zero | lptt 0 = absent | yes |
| all2all_default | allgather_reducescatter = absent | yes |
| all2all_alias | pplx = allgather_reducescatter | no |
| all2all_alias | naive = allgather_reducescatter | no |
| eplb_disabled | eplb off = absent | yes |
| eplb_disabled | eplb off, 8 redundant = absent | yes |
| eplb_zero_redundant | eplb on, 0 redundant = eplb on | yes |
| inert:window_size | eplb on, 8 redundant: window_size 1 = 1000000 | yes |
| inert:window_size | eplb on, 16 redundant: window_size 1 = 1000000 | yes |
| inert:step_interval | eplb on, 8 redundant: step_interval 1 = 1000000 | yes |
| inert:step_interval | eplb on, 16 redundant: step_interval 1 = 1000000 | yes |
| control | lptt 64 != absent | no |
| control | eplb on, 8 redundant != eplb off | no |

identical on every pair: long_prefill_token_threshold_zero, all2all_default, eplb_disabled, eplb_zero_redundant, inert:window_size, inert:step_interval
not identical: all2all_alias

## minimax-m2.7/h100/chatbot, v0.15.0

```
uv run python scripts/measure_folds.py v0.15.0 --model minimax-m2.7 --hardware h100
```

stack v0.15.0, engine 0.32.0, minimax-m2.7/h100/chatbot, 1 x tp4*dp2 + EP, c=64, 1280 requests, seed 0; base {'tp': 4, 'dp': 2, 'enable_expert_parallel': True, 'num_instances': 1, 'quantization': 'fp8', 'cache_dtype': 'fp8', 'concurrency': 64}

compared: BLIS stdout less name, runtime.wall_clock_s, runtime.started_utc, input.scenario.name, input.deployment.name, input.deployment.pools[*].engine

| fold / field | claim | identical |
|---|---|---|
| long_prefill_token_threshold_zero | lptt 0 = absent | yes |
| all2all_default | allgather_reducescatter = absent | yes |
| all2all_alias | pplx = allgather_reducescatter | no |
| all2all_alias | naive = allgather_reducescatter | no |
| eplb_disabled | eplb off = absent | yes |
| eplb_disabled | eplb off, 32 redundant = absent | yes |
| eplb_zero_redundant | eplb on, 0 redundant = eplb on | yes |
| inert:window_size | eplb on, 32 redundant: window_size 1 = 1000000 | yes |
| inert:window_size | eplb on, 64 redundant: window_size 1 = 1000000 | yes |
| inert:step_interval | eplb on, 32 redundant: step_interval 1 = 1000000 | yes |
| inert:step_interval | eplb on, 64 redundant: step_interval 1 = 1000000 | yes |
| control | lptt 64 != absent | no |
| control | eplb on, 32 redundant != eplb off | no |

identical on every pair: long_prefill_token_threshold_zero, all2all_default, eplb_disabled, eplb_zero_redundant, inert:window_size, inert:step_interval
not identical: all2all_alias

## Check: the script on v0.14.0 reproduces v0.14.0's claims

```
uv run python scripts/measure_folds.py v0.14.0 --model glm-5.3-flash --hardware h100   # BLIS_* = the v0.14.0 stack
```

stack v0.14.0, engine 0.29.0, glm-5.3-flash/h100/chatbot, 1 x tp4*dp2 + EP, c=64, 1280 requests, seed 0; base {'tp': 4, 'dp': 2, 'enable_expert_parallel': True, 'num_instances': 1, 'quantization': 'fp8', 'cache_dtype': 'fp8', 'concurrency': 64}

compared: BLIS stdout less name, runtime.wall_clock_s, runtime.started_utc, input.scenario.name, input.deployment.name, input.deployment.pools[*].engine

| fold / field | claim | identical |
|---|---|---|
| long_prefill_token_threshold_zero | lptt 0 = absent | yes |
| all2all_default | allgather_reducescatter = absent | yes |
| all2all_alias | pplx = allgather_reducescatter | yes |
| all2all_alias | naive = allgather_reducescatter | yes |
| eplb_disabled | eplb off = absent | yes |
| eplb_disabled | eplb off, 32 redundant = absent | yes |
| eplb_zero_redundant | eplb on, 0 redundant = eplb on | yes |
| inert:window_size | eplb on, 32 redundant: window_size 1 = 1000000 | yes |
| inert:window_size | eplb on, 64 redundant: window_size 1 = 1000000 | yes |
| inert:step_interval | eplb on, 32 redundant: step_interval 1 = 1000000 | yes |
| inert:step_interval | eplb on, 64 redundant: step_interval 1 = 1000000 | yes |
| control | lptt 64 != absent | no |
| control | eplb on, 32 redundant != eplb off | no |

identical on every pair: long_prefill_token_threshold_zero, all2all_default, all2all_alias, eplb_disabled, eplb_zero_redundant, inert:window_size, inert:step_interval
not identical: none

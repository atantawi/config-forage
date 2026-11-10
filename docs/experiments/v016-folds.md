# canonical_config folds and inert eplb fields, re-measured on BLIS v0.16.0

Stack v0.16.0: inference-sim v0.16.0 @ 3f4998e (blis-substrate v0.13.0, blis-vllm v0.32.0-blis.2,
router v1.0.0-blis.3), the catalog and registry embedded in the binary, `engine_version: "0.32.0"`.
Built natively on arm64 (Go 1.27.3). Script: `scripts/measure_folds.py`, the same protocol as
[v015-folds.md](v015-folds.md); output below verbatim. Nothing in the v0.16.0 tables
(`Stack.measured_folds`, `Stack.inert_block_fields` in `config_forage/blis/stacks.py`) is copied
from v0.15.0. Each entry was measured on this stack.

**Outcome.** The tables match v0.15.0's.
- The folds `long_prefill_token_threshold_zero`, `all2all_default`, `eplb_disabled` and
  `eplb_zero_redundant` are byte-identical on both models.
- `all2all_alias` is not identical: pplx and naive keep their own runs, because the Result records
  the engine's rename warning.
- The eplb fields `window_size` and `step_interval` changed. On v0.15.0 they were *inert* (accepted,
  not priced). On v0.16.0 **BLIS refuses them** when stated at anything but vLLM's default:
  ```
  engine vllm 0.32.0: vllm: eplb.window_size (vLLM's eplb_config.window_size) is not modelled: this
  engine and the cost model read no such setting, so a value other than vLLM's default would be dropped
  (https://github.com/inference-sim/blis-vllm/issues/76)
  ```
  This comes from blis-vllm v0.32.0-blis.2 `core/vllm/config.go:623-631`, where the defaults are
  window_size 1000 and step_interval 3000. A stated default runs (rc 0, checked with window_size
  1000); 7 for either is refused. config-forage refuses both fields before a run, as it did on v0.15.0,
  with the new reason. So the 1-vs-1000000 inert pairs cannot be run, and `--no-inert` skips them.

Both models and their settings are v0.15.0's. glm-5.3-flash still does not run on engine 0.32.0.

## kimi-k2.5/h200/chatbot, v0.16.0

```
uv run python scripts/measure_folds.py v0.16.0 --model kimi-k2.5 --hardware h200 --redundant 8,16 \
    --set '{"gpu_memory_utilization": 0.95}' --no-inert
```

stack v0.16.0, engine 0.32.0, kimi-k2.5/h200/chatbot, 1 x tp4*dp2 + EP, c=64, 1280 requests, seed 0; base {'tp': 4, 'dp': 2, 'enable_expert_parallel': True, 'num_instances': 1, 'quantization': 'fp8', 'cache_dtype': 'fp8', 'concurrency': 64, 'gpu_memory_utilization': 0.95}

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
| control | lptt 64 != absent | no |
| control | eplb on, 8 redundant != eplb off | no |

identical on every pair: long_prefill_token_threshold_zero, all2all_default, eplb_disabled, eplb_zero_redundant
not identical: all2all_alias

## minimax-m2.7/h100/chatbot, v0.16.0

```
uv run python scripts/measure_folds.py v0.16.0 --model minimax-m2.7 --hardware h100 --no-inert
```

stack v0.16.0, engine 0.32.0, minimax-m2.7/h100/chatbot, 1 x tp4*dp2 + EP, c=64, 1280 requests, seed 0; base {'tp': 4, 'dp': 2, 'enable_expert_parallel': True, 'num_instances': 1, 'quantization': 'fp8', 'cache_dtype': 'fp8', 'concurrency': 64}

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
| control | lptt 64 != absent | no |
| control | eplb on, 32 redundant != eplb off | no |

identical on every pair: long_prefill_token_threshold_zero, all2all_default, eplb_disabled, eplb_zero_redundant
not identical: all2all_alias

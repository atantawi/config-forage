# S-mn: glm-5.3 on 32 H100, chatbot, BLIS v0.14.0 — screen and ground truth

Stack: inference-sim v0.14.0 @ f2ba6bb (kernel v0.4.0, blis-vllm v0.32.0), blis-catalog 0.3.0,
blis-registry v0.2.0, `engine_version: "0.29.0"`, 1280 requests per point.

**Scenario:** glm-5.3 (MoE, 256 routed experts, 64 KV heads), fp8 weights, on 32 H100s, chatbot, ITL
p99 ≤ 20 ms, headroom 0.8. Score = tok/s per GPU at ⌊0.8·C*⌋.

**Multi-node is required.** At tp·dp = 8 the kernel reports 88.4 GiB of fixed occupancy per rank on a
79.2 GiB H100, whatever `max_model_len`, `gpu_memory_utilization` or KV dtype is set. So every
instance spans 2 nodes (width 16) or 4 nodes (width 32) over ib-400g. That gives 22 topologies.

## Screen (one axis at a time, 7 bases)

Screening space: `spaces/screen/glm53-h100-chatbot-32-screen.yaml`. Raw output:
`results/screen-mn/sensitivity.jsonl`, `results/screen-mn/summary.txt`. About 1,450 BLIS runs.

| base | tok/s/GPU | axes that matter (one-axis spread) |
|---|---|---|
| 2 × tp16 + EP (default) | 25.8 | EP 30%; kv weight 0 −11% |
| 2 × tp16 + EP, inflight-load | 25.9 | EP 30% |
| 2 × tp8·dp2 + EP, ext LB, inflight-load | 19.7 | EP 34%; router family 13%; active weight 4% |
| 2 × tp8·dp2 + EP | 17.6 | EP 61%; router family 11%; DP LB 3% |
| 2 × tp4·dp4 + EP | 14.9 | EP 68%; **DP LB 49%** (external fails); router family 10% |
| 1 × tp32 + EP (4 nodes) | 13.9 | EP 35% |
| 1 × tp8·dp4 + EP (4 nodes) | 9.9 | EP 29%; DP LB 7% |

**Under 1% at every base**, so pinned in S-mn:
- `cache_dtype`: max 0.9%.
- `gpu_memory_utilization`, `block_size` (64/128), `scheduler`, `preemption_policy`: 0%.
- `max_num_batched_tokens`: max 0.4%.
- `max_num_seqs`: max 0.6%. This differs from S-mid v3, where it cost up to 43% at 1 × tp8·dp2.
- Queue weight: 0%.

## Ground truth

`results/s-mn/grid.jsonl`: 960 configs (22 topologies × router family and weights × DP LB). 4,178
unique BLIS runs, 2 h 26 min at 9 workers. 576 configs score and none is censored. The other 384
fail at concurrency 1: every config of tp1·dp16, tp1·dp32, tp2·dp8 and tp2·dp16, with ITL p99 ≈ 30
ms at c = 1. A single c = 1 run of those dp-heavy shapes takes 100–250 s of wall time.

**Optimum: 25.94 tok/s per GPU** at 2 × tp16 + EP, inflight-load, kv weight 1, active weight 1.

| topology (best config) | tok/s/GPU | gap |
|---|---|---|
| 2 × tp16 + EP | 25.94 | — |
| 2 × tp8·dp2 + EP | 19.70 | −24.0% |
| 2 × tp16, no EP | 18.12 | −30.1% |
| 2 × tp4·dp4 + EP | 16.64 | −35.8% |
| 1 × tp32 + EP | 13.85 | −46.6% |
| 2 × tp8·dp2, no EP | 13.07 | −49.6% |
| 1 × tp16·dp2 + EP | 10.67 | −58.9% |
| 1 × tp8·dp4 + EP | 9.89 | −61.9% |
| 8 topologies (tp ≤ 2) | fail | — |

- **Within 1%:** 21 configs, all of them 2 × tp16 + EP. That is 21 of that topology's 24 configs.
  The 3 below are exactly the kv-weight-0 configs (−11%).
- **Within 0.5%:** 12 configs.
- **The 1% plateau ends in a cliff.** Nothing scores between −1% and −11% (within 2% and within 5%
  are also 21 configs).
- **The default** (2 × tp16 + EP, load-only) scores 25.81, 0.51% from the optimum. The default
  topology was chosen before the screen, as the narrowest that fits. So this closeness is not
  independent evidence.

## Compared with S-mid v3 (glm-5.3-flash, 16 H100)

| | S-mid v3 | S-mn |
|---|---|---|
| best topology | 2 × tp8 + EP | 2 × tp16 + EP |
| gap to next topology | 16% | 24% |
| configs within 1% | 105, all one topology | 21, all one topology |
| plateau's lower edge | kv weight 0 (−12%) | kv weight 0 (−11%) |
| EP effect | 9–58% | 29–68% |
| `max_num_seqs` | up to 43% (dp > 1, multi-node) | under 0.6% everywhere |
| engine knobs that matter | `max_num_seqs` | none |
| router family | 0–12% | 0–13% (dp > 1 only) |
| infeasible at c = 1 | none | 8 of 22 topologies (tp ≤ 2) |

The structure is the same: one narrowest-fitting tp + EP topology wins, with a flat plateau and
cliffs everywhere else, and the engine knobs are flat. Multi-node makes the cliffs steeper.

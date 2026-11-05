"""Stack records: a space without `stack` is BLIS v0.14.0 and keeps every hash and cache key it had."""

import json
import os
import re
from pathlib import Path

import pytest

from config_forage import router
from config_forage.blis import run_blis, workloads
from config_forage.blis import runner
from config_forage.blis.runner import cache_key
from config_forage.blis.stacks import STACKS, stack_of
from config_forage.space import Scenario, load_space, space_hash

ROOT = Path(__file__).resolve().parent.parent
CHAT = ROOT / "spaces" / "glm53flash-h100-chatbot-real-16.yaml"

# Captured on main (704c7ad) before Scenario gained `stack`.
SPACE_HASHES = {
    "spaces/glm53-h100-chatbot-32.yaml": "491ce35fce766d99de4f62829a7b8c2166518e013fa2547f75b41e832a68dd8f",
    "spaces/glm53flash-h100-chatbot-16-large.yaml": "74d887a2fd4a1bfa1d134c20346f9d85f2cbc0a6d403df2c06896130bdd80027",
    "spaces/glm53flash-h100-chatbot-16.yaml": "0d719dc03ae200e882d7b9a5d3ca78f8d538363f674bafe3610bdb369dbb5732",
    "spaces/glm53flash-h100-chatbot-real-16-large.yaml": "b6597584e4beb94d8e0d9772d3d94bae3f324371fc95214cdee5bf78d06e2946",
    "spaces/glm53flash-h100-chatbot-real-16.yaml": "9fd580ae30daa97e5e21cb210e8f4a7b3691e1d955105ecaea3d76f36af587a4"
}

CONFIGS = [{"tp": 1, "dp": 8, "enable_expert_parallel": False, "data_parallel_load_balancing": "internal", "quantization": "fp8", "cache_dtype": "auto", "gpu_memory_utilization": 0.9, "max_num_batched_tokens": 8192, "block_size": 64, "max_num_seqs": 32, "endpoint_picker_config": "load-only", "kv_cache_utilization_scorer_weight": 0, "num_instances": 2}, {"tp": 1, "dp": 8, "enable_expert_parallel": False, "data_parallel_load_balancing": "internal", "quantization": "fp8", "cache_dtype": "auto", "gpu_memory_utilization": 0.9, "max_num_batched_tokens": 8192, "block_size": 64, "max_num_seqs": 32, "endpoint_picker_config": "load-only", "kv_cache_utilization_scorer_weight": 4, "num_instances": 2}, {"tp": 1, "dp": 8, "enable_expert_parallel": False, "data_parallel_load_balancing": "internal", "quantization": "fp8", "cache_dtype": "auto", "gpu_memory_utilization": 0.9, "max_num_batched_tokens": 8192, "block_size": 64, "max_num_seqs": 32, "endpoint_picker_config": "load-only", "kv_cache_utilization_scorer_weight": 2, "num_instances": 2}]
# (index into CONFIGS, concurrency) -> key; captured on main before the change, no workload shape.
KEYS = {(0, 1): "2abccbe571f5b8c016fbd13940c688dc1b9d332b7cf3de69195e0a67f7bf2898", (0, 64): "e63b8519753d360f80b4cfda26e518d01a3e642d7b17d2d19624596641e0875c", (1, 1): "0eeb80fe8fa807a0285e75896384a0c480e97cb05c9a7dda3aac53eeb62ec573", (1, 64): "e2c8dce171cfec4602adebfa1cce12cd7ef78abc4aa9d3de9e7e2743c4d912d8", (2, 1): "143a51008530df2ab2b4868f3197b8820c3066984a69bba3576a2f9c7da87873", (2, 64): "339fc7c9295a3cb2314d1c219933c293fa79aed7e8d66810552e7ea0dc84ec5f"}


def _key(sc, i, c):
    cfg = run_blis.canonical_config(router.to_blis(CONFIGS[i]), sc.engine_version)
    return cache_key(cfg, c, sc.seed, sc)


def test_v014_pins_unchanged():
    assert STACKS["v0.14.0"].pins == (
        "inference-sim v0.14.0 f2ba6bb1c189ed5a56933cfa3d0927eae3836ee9", "kernel v0.4.0",
        "blis-vllm v0.32.0", "catalog 0.3.0", "registry v0.2.0", "workload as token-shape flags")
    assert runner.PINS is STACKS["v0.14.0"].pins


def test_absent_stack_is_v014():
    assert stack_of(load_space(CHAT).scenario).name == "v0.14.0"


@pytest.mark.parametrize("name", sorted(SPACE_HASHES))
def test_space_hash_unchanged_for_existing_spaces(name):
    assert space_hash(load_space(ROOT / name)) == SPACE_HASHES[name]


@pytest.mark.parametrize("i,c", sorted(KEYS))
def test_cache_key_unchanged(i, c):
    assert _key(load_space(CHAT).scenario, i, c) == KEYS[(i, c)]


def test_v014_keys_hit_real_cache():
    grid = Path(os.environ.get("CF_GRID_FOR_CACHE_TEST", ROOT / "results" / "s-chat" / "grid.jsonl"))
    catalog = Path(os.environ.get("BLIS_CATALOG_V014", ROOT.parent / "blis-stack-v014" / "blis-catalog"))
    cache = ROOT / "cache"
    if not (grid.exists() and cache.is_dir() and catalog.is_dir()):
        pytest.skip("needs the S-chat grid, the v0.14.0 catalog and the cache")
    meta = json.loads(Path(f"{grid}.meta.json").read_text())
    sp = load_space(CHAT)
    if meta["space_hash"] != space_hash(sp):
        pytest.skip("grid was made from another space")
    shape = workloads.shape(sp.scenario.workload, catalog)
    with grid.open() as f:
        rows = [json.loads(next(f)) for _ in range(20)]
    for r in rows:
        cfg = run_blis.canonical_config(router.to_blis(r["config"]), sp.scenario.engine_version)
        k = cache_key(cfg, r["c_use"], sp.scenario.seed, sp.scenario, shape)
        assert (cache / k[:2] / f"{k}.json").exists()


def test_v015_differs():
    sc = load_space(CHAT).scenario
    v15 = Scenario(**{**sc.__dict__, "stack": "v0.15.0"})
    assert stack_of(v15).name == "v0.15.0"
    assert _key(v15, 0, 1) != _key(sc, 0, 1)
    assert space_hash(load_space(CHAT)) != space_hash(
        type(load_space(CHAT))(v15, load_space(CHAT).knobs))


def test_unknown_stack_refused():
    sc = Scenario(**{**load_space(CHAT).scenario.__dict__, "stack": "v0.13.0"})
    with pytest.raises(ValueError, match="v0.14.0.*v0.15.0|v0.15.0.*v0.14.0"):
        stack_of(sc)


# --- per-stack runner tables -----------------------------------------------------------------------

V14, V15 = STACKS["v0.14.0"], STACKS["v0.15.0"]
EP = {"tp": 4, "dp": 2, "enable_expert_parallel": True}
# scenario_yaml on main (704c7ad) for the input below, before it took `coefficients`.
V014_YAML = 'kind: Scenario\nname: n\nengine_version: "0.29.0"\nmodel: glm-5.3-flash\ncoefficients: [cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-recurrent, cost-model-memory]\ncluster:\n  hardware: h100\n  fabric: ib-400g\n  nodes: 2\n  gpus_per_node: 8\n---\nkind: Deployment\nname: n\npools:\n  - role: colocated\n    nodes: 2\n    parallel:\n      tp: 4\n      pp: 1\n      dp: 2\n      enable_expert_parallel: true\n    engine:\n      max_num_batched_tokens: 8192\n      max_num_seqs: 256\n      max_model_len: 32768\n      cudagraph_mode: "PIECEWISE"\n      gpu_memory_utilization: 0.9\n      quantization: "fp8"\n      all2all_backend: "deepep_v2"\n      eplb:\n        enable_eplb: true\n        num_redundant_experts: 32\n      block_size: 64\nrouter:\n  endpoint_picker_config: epp-x.yaml\n'
V014_SCENARIO_FILE = "glm-5.3-flash-h100-680dd34160.yaml"   # write_scenario's name on main


def _deployment(stack=V14, engine_version="0.29.0"):
    p, e, _, _ = run_blis.classify_config(
        {**EP, "quantization": "fp8", "all2all_backend": "deepep_v2",
         "eplb": {"enable_eplb": True, "num_redundant_experts": 32}}, engine_version, stack)
    e["block_size"] = 64
    return p, e


def test_stack_pins_name_its_commit_and_tags():
    for s in STACKS.values():
        assert s.pins[0].split()[-1] == s.sim_sha
        assert f"catalog {s.catalog_tag}" in s.pins and f"registry {s.registry_tag}" in s.pins


def test_v015_scenario_has_fabric_coefficients(tmp_path):
    p, e = _deployment()
    assert run_blis.scenario_yaml("glm-5.3-flash", "h100", "0.29.0", p, e, "n", num_instances=2,
                                  epp_file="epp-x.yaml") == V014_YAML
    assert run_blis.scenario_yaml("glm-5.3-flash", "h100", "0.29.0", p, e, "n", num_instances=2,
                                  epp_file="epp-x.yaml", coefficients=V14.coefficients) == V014_YAML
    assert run_blis.write_scenario(tmp_path, "glm-5.3-flash", "h100", "0.29.0", p, e, 2,
                                   epp_file="epp-x.yaml") == V014_SCENARIO_FILE
    y15 = run_blis.scenario_yaml("glm-5.3-flash", "h100", "0.29.0", p, e, "n", num_instances=2,
                                 epp_file="epp-x.yaml", coefficients=V15.coefficients)
    assert "cost-model-fabric]" in y15 and y15 != V014_YAML
    # the same deployment under other coefficients never reuses the v0.14.0 file
    assert run_blis.write_scenario(tmp_path, "glm-5.3-flash", "h100", "0.29.0", p, e, 2, epp_file="epp-x.yaml",
                                   coefficients=V15.coefficients) != V014_SCENARIO_FILE


def test_v015_all2all_table():
    rel = run_blis.ENGINE_RELEASES["0.32.0"]
    assert set(rel["all2all_backends"]) == {
        "naive", "allgather_reducescatter", "pplx", "deepep_high_throughput", "deepep_low_latency",
        "deepep_v2", "flashinfer_all2allv", "flashinfer_nvlink_one_sided", "flashinfer_nvlink_two_sided",
        "mori_high_throughput", "mori_low_latency", "moonep", "nixl_ep", "passthrough"}
    assert rel["all2all_default"] == "allgather_reducescatter"
    assert rel["all2all_aliases"] == {"pplx": "allgather_reducescatter", "naive": "allgather_reducescatter"}
    with pytest.raises(run_blis.ConfigError, match=r"not accepted by engine 0\.32\.0"):
        run_blis.classify_config({**EP, "all2all_backend": "bogus"}, "0.32.0", V15)
    assert run_blis.classify_config({**EP, "all2all_backend": "moonep"}, "0.32.0", V15)[1]["all2all_backend"] == "moonep"
    with pytest.raises(run_blis.ConfigError, match="not accepted by engine 0.29.0"):
        run_blis.classify_config({**EP, "all2all_backend": "moonep"}, "0.29.0", V14)


def test_v014_keeps_every_fold():
    assert V14.measured_folds == frozenset(run_blis.FOLDS)
    assert V14.inert_block_fields is run_blis.UNSUPPORTED_BLOCK_FIELDS


def test_v015_tables_as_measured():
    """docs/experiments/v015-folds.md: every fold but all2all_alias, and both eplb fields, identical."""
    assert V15.measured_folds == frozenset(run_blis.FOLDS) - {"all2all_alias"}
    ep = {**EP, "num_instances": 1}
    # an alias stays as written (its own run); the default spelling still folds away
    for a in ("pplx", "naive"):
        assert run_blis.canonical_config({**ep, "all2all_backend": a}, "0.32.0", V15) == {**ep, "all2all_backend": a}
    assert run_blis.canonical_config({**ep, "all2all_backend": "allgather_reducescatter"}, "0.32.0", V15) == ep
    assert run_blis.canonical_config({**ep, "long_prefill_token_threshold": 0}, "0.32.0", V15) == ep
    assert set(V15.inert_block_fields["eplb"]) == {"window_size", "step_interval"}
    assert all("v0.15.0" in r for r in V15.inert_block_fields["eplb"].values())


NOOPS = [{"long_prefill_token_threshold": 0}, {"all2all_backend": "allgather_reducescatter"},
         {"all2all_backend": "pplx"}, {"all2all_backend": "naive"}, {"eplb": {"enable_eplb": False}},
         {"eplb": {"enable_eplb": True, "num_redundant_experts": 0}}]


@pytest.mark.parametrize("noop", NOOPS)
def test_v015_no_unmeasured_folds(noop):
    """A fold applies on v0.15.0 only once it is listed in measured_folds (re-measured there)."""
    from dataclasses import replace
    cfg = {**EP, **noop}
    folded = run_blis.canonical_config(cfg, "0.32.0", replace(V15, measured_folds=frozenset(run_blis.FOLDS)))
    assert folded != cfg                                                      # each one is a fold
    assert run_blis.canonical_config(cfg, "0.32.0", replace(V15, measured_folds=frozenset())) == cfg
    assert run_blis.canonical_config(cfg, "0.29.0") == run_blis.canonical_config(cfg, "0.29.0", V14) == folded


def test_inert_block_fields_are_per_stack():
    from dataclasses import replace
    cfg = {**EP, "eplb": {"enable_eplb": True, "window_size": 1000}}
    with pytest.raises(run_blis.ConfigError, match="inert in BLIS v0.14.0"):
        run_blis.classify_config(cfg, "0.29.0", V14)
    assert run_blis.classify_config(cfg, "0.32.0", replace(V15, inert_block_fields={}))[1]["eplb"] == cfg["eplb"]
    with pytest.raises(run_blis.ConfigError, match="window_size.*must be a non-negative integer"):
        run_blis.classify_config({**EP, "eplb": {"enable_eplb": True, "window_size": -1}}, "0.32.0",
                                 replace(V15, inert_block_fields={}))


def _git_repo(d):
    import subprocess
    d.mkdir(parents=True)
    run = lambda *a: subprocess.run(["git", "-C", str(d), "-c", "tag.gpgsign=false", "-c", "commit.gpgsign=false",
                                     *a], check=True, capture_output=True, text=True).stdout
    run("init", "-q")
    (d / "main.go").write_text("package main\n")
    run("add", "main.go")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
    return run("rev-parse", "HEAD").strip(), run


def test_stack_mismatch_refused(tmp_path):
    sha, _ = _git_repo(tmp_path / "sim")
    (tmp_path / "sim" / "blis").write_text("")
    env = runner.BlisEnv(tmp_path / "sim" / "blis", tmp_path / "cat", tmp_path / "reg", tmp_path / "sc")
    with pytest.raises(ValueError) as e:
        runner.check_binary(env, V15)
    assert sha in str(e.value) and V15.sim_sha in str(e.value)
    sc = Scenario(**{**load_space(CHAT).scenario.__dict__, "stack": "v0.15.0", "engine_version": "0.32.0"})
    with pytest.raises(ValueError, match=V15.sim_sha):
        runner.CachedRunner(env, sc, tmp_path / "cache")


def test_stack_tags_checked(tmp_path):
    from dataclasses import replace
    sha, _ = _git_repo(tmp_path / "sim")
    _, cat = _git_repo(tmp_path / "cat")
    _, reg = _git_repo(tmp_path / "reg")
    cat("tag", V15.catalog_tag)
    env = runner.BlisEnv(tmp_path / "sim" / "blis", tmp_path / "cat", tmp_path / "reg", tmp_path / "sc")
    st = replace(V15, sim_sha=sha)
    with pytest.raises(ValueError, match=f"registry.*{V15.registry_tag}"):
        runner.check_binary(env, st)
    reg("tag", "v9.9.9")
    with pytest.raises(ValueError, match="v9.9.9"):
        runner.check_binary(env, st)
    reg("tag", "-d", "v9.9.9")
    reg("tag", V15.registry_tag)
    runner.check_binary(env, st)                       # all three match


def test_binary_outside_a_checkout_warns(tmp_path, capsys):
    _, cat = _git_repo(tmp_path / "cat")
    _, reg = _git_repo(tmp_path / "reg")
    cat("tag", V15.catalog_tag)
    reg("tag", V15.registry_tag)
    env = runner.BlisEnv(tmp_path / "blis", tmp_path / "cat", tmp_path / "reg", tmp_path / "sc")
    runner.check_binary(env, V15)
    assert "not inside" in capsys.readouterr().err


def test_binary_outside_a_checkout_still_checks_catalog_and_registry(tmp_path, capsys):
    """A copied binary (e.g. an out-of-tree amd64 build) cannot be checked, but the data it runs against can:
    a catalog or registry at another tag is refused, as it is for a binary inside a checkout."""
    _, cat = _git_repo(tmp_path / "cat")
    _, reg = _git_repo(tmp_path / "reg")
    env = runner.BlisEnv(tmp_path / "blis", tmp_path / "cat", tmp_path / "reg", tmp_path / "sc")
    with pytest.raises(ValueError, match=f"catalog.*{V15.catalog_tag}"):
        runner.check_binary(env, V15)
    assert "not inside" in capsys.readouterr().err   # the binary warning is still given
    cat("tag", V15.catalog_tag)
    with pytest.raises(ValueError, match=f"registry.*{V15.registry_tag}"):
        runner.check_binary(env, V15)
    reg("tag", V15.registry_tag)
    runner.check_binary(env, V15)                      # catalog and registry match: warning only


def test_runner_passes_its_stack_to_run_one(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(run_blis, "run_one", lambda *a, **kw: seen.update(kw) or {"returncode": 0, "metrics": {}, "error": None})
    monkeypatch.setattr(runner, "check_binary", lambda env, stack: None)
    monkeypatch.setattr(workloads, "shape", lambda w, c: {"x": 1})
    sc = Scenario(**{**load_space(CHAT).scenario.__dict__, "stack": "v0.15.0", "engine_version": "0.32.0"})
    env = runner.BlisEnv(tmp_path / "blis", tmp_path / "cat", tmp_path / "reg", tmp_path / "sc")
    runner.CachedRunner(env, sc, tmp_path / "cache").run({"tp": 8}, 8)
    assert seen["stack"] is V15


def test_grid_refuses_other_stack(tmp_path):
    from test_bench import TINY, fake_blis

    from config_forage.bench import grid
    sp = load_space(TINY)
    sp15 = type(sp)(Scenario(**{**sp.scenario.__dict__, "stack": "v0.15.0", "engine_version": "0.32.0"}), sp.knobs)
    out = tmp_path / "grid.jsonl"
    grid.grid(sp, runner.CachedRunner(None, sp.scenario, tmp_path / "c", run_fn=fake_blis), out, workers=2, log=None)
    assert json.loads(grid.meta_path(out).read_text())["pins"] == list(V14.pins)
    with pytest.raises(ValueError, match="pins"):
        grid.grid(sp15, runner.CachedRunner(None, sp15.scenario, tmp_path / "c", run_fn=fake_blis), out,
                  workers=2, log=None)


# --- a stack runs its one engine release ------------------------------------------------------------

@pytest.mark.parametrize("ev, stack", [("0.32.0", V14), ("0.29.0", V15), ("0.31.0", V15)])
def test_engine_version_other_than_the_stacks_refused(tmp_path, ev, stack):
    with pytest.raises(run_blis.ConfigError, match=f"{re.escape(ev)}.*{stack.name}.*{re.escape(stack.engine_version)}"):
        run_blis.classify_config(EP, ev, stack)
    with pytest.raises(run_blis.ConfigError, match="a stack runs one engine release"):
        run_blis.canonical_config({**EP, "long_prefill_token_threshold": 0}, ev, stack)
    with pytest.raises(run_blis.ConfigError, match="a stack runs one engine release"):
        run_blis.run_one("blis", None, None, str(tmp_path), "m", "h100", ev, None, 1, 0, None, EP, stack=stack)
    sc = Scenario(**{**load_space(CHAT).scenario.__dict__, "stack": stack.name, "engine_version": ev})
    with pytest.raises(ValueError, match=f"{re.escape(ev)}.*{stack.name}"):
        runner.CachedRunner(None, sc, tmp_path / "cache")


def test_default_stack_is_v014_at_0_29_0():
    assert run_blis.classify_config(EP)[0]["tp"] == 4                       # 0.29.0 on v0.14.0
    with pytest.raises(run_blis.ConfigError, match="v0.14.0"):
        run_blis.classify_config(EP, "0.32.0")                              # no stack named: v0.14.0


@pytest.mark.parametrize("path", sorted(str(p.relative_to(ROOT)) for p in (ROOT / "spaces").rglob("*.yaml")))
def test_every_space_states_its_stacks_engine_version(path):
    sc = load_space(ROOT / path).scenario
    assert sc.engine_version == stack_of(sc).engine_version


def test_blis_env_for_skips_another_commit_and_fails_a_wrong_tag(tmp_path, monkeypatch):
    from dataclasses import replace

    from helpers import blis_env_for
    sha, _ = _git_repo(tmp_path / "sim")
    _git_repo(tmp_path / "cat")[1]("tag", V15.catalog_tag)
    _git_repo(tmp_path / "reg")[1]("tag", "v9.9.9")
    for k, d in (("BLIS_BIN", "sim/blis"), ("BLIS_CATALOG", "cat"), ("BLIS_REGISTRY", "reg"), ("BLIS_SCENARIOS", "sc")):
        monkeypatch.setenv(k, str(tmp_path / d))
    with pytest.raises(pytest.skip.Exception):                          # another commit: skip
        blis_env_for("v0.15.0")
    monkeypatch.setitem(STACKS, "v0.15.0", replace(V15, sim_sha=sha))
    with pytest.raises(ValueError, match="registry.*v9.9.9"):            # right commit, wrong tag: fail
        blis_env_for("v0.15.0")

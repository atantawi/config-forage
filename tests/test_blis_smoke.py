"""Real-BLIS smoke test: one glm-5.3-flash/h100/chatbot run and one probe on v0.14.0, one
kimi-k2.5/h200/chatbot run on v0.15.0. Skipped without BLIS_BIN, and each skipped unless BLIS_* is its stack."""

import os

import pytest
from helpers import blis_env_for

from config_forage.blis.runner import CachedRunner
from config_forage.probe import probe
from config_forage.space import Scenario, Slo

pytestmark = [pytest.mark.blis,
              pytest.mark.skipif(not os.environ.get("BLIS_BIN"), reason="BLIS_BIN not set")]

SCENARIO = Scenario("glm-5.3-flash", "h100", "chatbot", 16, "0.29.0", Slo(20.0),
                    model_entry={"arch": "MoE", "heads_kv": 64})
CONFIG = {"tp": 8, "enable_expert_parallel": True, "num_instances": 2,
          "quantization": "fp8", "cache_dtype": "fp8"}


@pytest.fixture
def runner(tmp_path):
    env = blis_env_for("v0.14.0")
    env.scenarios_dir = tmp_path / "scenarios"
    return CachedRunner(env, SCENARIO, tmp_path / "cache")


def test_one_run(runner):
    r = runner.run(CONFIG, 8)
    assert r["error"] is None, r.get("stderr_tail")
    assert r["metrics"]["tokens_per_sec"] > 0
    assert r["deployment"]["gpus"] == 16


def test_probe_scores(runner):
    p = probe(runner, CONFIG, SCENARIO)
    assert p.score is not None and p.score < 0
    assert p.c_star >= 1 and p.runs >= 1


def test_workload_reaches_blis(tmp_path):
    """Closed-loop BLIS ignores --workload; the runner's shape flags must make the preset bite."""
    env = blis_env_for("v0.14.0")
    env.scenarios_dir = tmp_path / "scenarios"
    per_req = {}
    for w in ("chatbot", "distribution"):
        r = CachedRunner(env, Scenario(**{**SCENARIO.__dict__, "workload": w}), tmp_path / w).run(CONFIG, 8)
        assert r["error"] is None, r.get("stderr_tail")
        per_req[w] = r["metrics"]["total_input_tokens"] / r["metrics"]["injected_requests"]
    assert 230 < per_req["chatbot"] < 280 and 480 < per_req["distribution"] < 560  # means 256 and 512


def test_catalog_presets_pass_the_bounds():
    """Every shape preset in the pinned catalog resolves (the bounds match what BLIS accepts)."""
    from pathlib import Path

    from config_forage.blis import workloads
    names = sorted(p.stem for p in (Path(os.environ["BLIS_CATALOG"]) / "workloads").glob("*.yaml"))
    shapes = [n for n in names if "replay:" not in (Path(os.environ["BLIS_CATALOG"]) / "workloads" / f"{n}.yaml").read_text()]
    assert {"chatbot", "multidoc"} <= set(shapes)
    for n in shapes:
        workloads.shape(n, os.environ["BLIS_CATALOG"])



# glm-5.3-flash does not run on engine 0.32.0 (docs/experiments/v015-folds.md): kimi-k2.5 on H200 does
V015 = Scenario("kimi-k2.5", "h200", "chatbot", 16, "0.32.0", Slo(50.0), stack="v0.15.0",
                model_entry={"arch": "MoE", "heads_kv": 64})


def test_v015_smoke(tmp_path):
    env = blis_env_for("v0.15.0")
    env.scenarios_dir = tmp_path / "scenarios"
    r = CachedRunner(env, V015, tmp_path / "cache").run(
        {"tp": 4, "dp": 2, "enable_expert_parallel": True, "num_instances": 1, "quantization": "fp8",
         "cache_dtype": "fp8"}, 64)
    # v0.15.0 prints a Result document; result_doc reads it into the v0.14.0 flat keys
    assert r["returncode"] == 0 and r["error"] is None, r.get("stderr_tail")
    m = r["metrics"]
    assert m["tokens_per_sec"] > 0 and m["itl_p99_ms"] > 0 and m["instance_id"] == "cluster"
    assert m["completed_requests"] == m["injected_requests"] == 256
    assert "cost-model-fabric" in (tmp_path / "scenarios" / r["deployment"]["scenario"]).read_text()


def test_ladder_example_default_scores(tmp_path):
    """The shipped ladder example runs end to end on v0.15.0 (so its model_entry and knobs cannot rot)."""
    from pathlib import Path

    from config_forage.probe import score_config
    from config_forage.space import load_space
    sp = load_space(Path(__file__).parent.parent / "spaces" / "examples" / "ladder-example.yaml")
    env = blis_env_for("v0.15.0")
    env.scenarios_dir = tmp_path / "scenarios"
    p = score_config(CachedRunner(env, sp.scenario, tmp_path / "cache"), sp.default_config(), sp.scenario)
    assert p.score is not None and p.c_star is None and p.diagnostics["scoring"] == "ladder"

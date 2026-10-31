"""Real-BLIS smoke test: one glm-5.3-flash/h100/chatbot run and one probe. Skipped without BLIS_BIN."""

import os

import pytest

from config_forage.blis.runner import BlisEnv, CachedRunner
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
    env = BlisEnv.from_env()
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
    env = BlisEnv.from_env()
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


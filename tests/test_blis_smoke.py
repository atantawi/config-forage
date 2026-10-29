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

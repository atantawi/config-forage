"""Router families from EndpointPickerConfig files named in a space (scenario.router_files)."""

import pytest
import yaml

from config_forage.blis.runner import CachedRunner
from config_forage.space import load_space, space_hash

EPP_A = {"apiVersion": "llm-d.ai/v1", "kind": "EndpointPickerConfig",
         "plugins": [{"type": "single-profile-handler"}, {"type": "random-picker"}],
         "schedulingProfiles": [{"name": "default", "plugins": [{"pluginRef": "random-picker"}]}]}
EPP_B = {"apiVersion": "llm-d.ai/v1", "kind": "EndpointPickerConfig",
         "plugins": [{"type": "single-profile-handler"}, {"type": "queue-scorer"}, {"type": "max-score-picker"}],
         "schedulingProfiles": [{"name": "default", "plugins": [{"pluginRef": "queue-scorer", "weight": 1},
                                                                {"pluginRef": "max-score-picker"}]}]}
LADDER_EXAMPLE_HASH = "9bdea35c0dff9d1460ab6a15dc6184bfc8e1a6b1f07961d8a38a5286dc22cd9d"


def _space(tmp_path, router_files=None, extra_knobs=""):
    d = tmp_path / "sp"
    (d / "routing").mkdir(parents=True, exist_ok=True)
    (d / "routing" / "a.yaml").write_text(yaml.safe_dump(EPP_A))
    (d / "routing" / "b.yaml").write_text(yaml.safe_dump(EPP_B))
    rf = router_files if router_files is not None else {"fam-a": "routing/a.yaml", "fam-b": "routing/b.yaml"}
    (d / "space.yaml").write_text(f"""
scenario:
  model: m
  hardware: h100
  workload: chatbot
  gpu_budget: 8
  engine_version: "0.29.0"
  router_files: {yaml.safe_dump(rf, default_flow_style=True).strip()}
knobs:
  - {{name: tp, values: [8]}}
  - {{name: endpoint_picker_config, values: [{", ".join(rf)}]}}
{extra_knobs}""")
    return d / "space.yaml"


def _fake(seen):
    def fn(config, concurrency, seed):
        seen.append(dict(config))
        return {"metrics": {"itl_p99_ms": 1.0}, "error": None, "returncode": 0, "wall_s": 0.1, "deployment": {}}
    return fn


def test_runner_substitutes_epp_dict(tmp_path):
    sp = load_space(_space(tmp_path))
    assert sp.scenario.router_families == {"fam-a": EPP_A, "fam-b": EPP_B}
    seen = []
    r = CachedRunner(None, sp.scenario, tmp_path / "cache", run_fn=_fake(seen))
    r.run(sp.config((0, 0)), 8)
    r.run(sp.config((0, 1)), 8)
    assert [s["epp_config"] for s in seen] == [EPP_A, EPP_B]
    assert all("endpoint_picker_config" not in s for s in seen)


def test_same_cache_key_as_explicit_epp_config(tmp_path):
    sp = load_space(_space(tmp_path))
    r = CachedRunner(None, sp.scenario, tmp_path / "cache", run_fn=_fake([]))
    k_family = r.run(sp.config((0, 0)), 8)["cache_key"]
    k_explicit = r.run({"tp": 8, "num_instances": 1, "epp_config": EPP_A}, 8)["cache_key"]
    assert k_family == k_explicit


def test_paths_relative_to_space_file(tmp_path, monkeypatch):
    path = _space(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert load_space(path.resolve()).scenario.router_families["fam-a"] == EPP_A


def test_name_collides_with_builtin(tmp_path):
    with pytest.raises(ValueError, match="load-only"):
        load_space(_space(tmp_path, {"load-only": "routing/a.yaml"}))


def test_weight_knob_with_file_family(tmp_path):
    sp = load_space(_space(tmp_path, extra_knobs="  - {name: queue_scorer_weight, values: [1, 2]}"))
    r = CachedRunner(None, sp.scenario, tmp_path / "cache", run_fn=_fake([]))
    with pytest.raises(ValueError, match="queue_scorer_weight"):
        r.run(sp.config((0, 0, 0)), 8)


def test_missing_file(tmp_path):
    with pytest.raises(ValueError, match="routing/nope.yaml"):
        load_space(_space(tmp_path, {"fam-a": "routing/nope.yaml"}))


def test_hash_changes_with_file_contents(tmp_path):
    path = _space(tmp_path)
    h = space_hash(load_space(path))
    (path.parent / "routing" / "a.yaml").write_text(yaml.safe_dump({**EPP_A, "plugins": EPP_A["plugins"][:1]}))
    assert space_hash(load_space(path)) != h


def test_hash_unchanged_without_router_files():
    assert space_hash(load_space("spaces/examples/ladder-example.yaml")) == LADDER_EXAMPLE_HASH

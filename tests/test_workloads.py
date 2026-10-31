"""Workload shapes: a named workload reaches BLIS as token-shape flags, never as --workload."""

import shutil

import pytest
import yaml

from config_forage.blis import run_blis, workloads
from config_forage.blis.runner import BlisEnv, CachedRunner, cache_key
from config_forage.space import Scenario

FALSE = shutil.which("false")  # exits 1: the command is built and recorded, BLIS is not needed

CHATBOT = """_comment: "provenance"
prefix_tokens: 0
prompt: {tokens: 256, tokens_stdev: 100, tokens_min: 2, tokens_max: 800}
output: {tokens: 256, tokens_stdev: 100, tokens_min: 1, tokens_max: 1024}
"""
MULTIDOC = """prompt: {tokens: 10240, tokens_stdev: 1200, tokens_min: 500, tokens_max: 20480}
output: {tokens: 1536, tokens_stdev: 300, tokens_min: 50, tokens_max: 4096}
"""


@pytest.fixture
def catalog(tmp_path):
    w = tmp_path / "catalog" / "workloads"
    w.mkdir(parents=True)
    (w / "chatbot.yaml").write_text(CHATBOT)
    (w / "multidoc.yaml").write_text(MULTIDOC)
    (w / "agentx-mvp.yaml").write_text("replay: {scenario: x}\n")
    (w / "odd.yaml").write_text(CHATBOT + "rate: 3\n")
    (w / "short.yaml").write_text("prompt: {tokens: 1}\noutput: {tokens: 1}\n")
    return tmp_path / "catalog"


def test_preset_shape(catalog):
    s = workloads.shape("chatbot", catalog)
    assert s == {"prefix_tokens": 0,
                 "prompt": {"tokens": 256, "tokens_stdev": 100, "tokens_min": 2, "tokens_max": 800},
                 "output": {"tokens": 256, "tokens_stdev": 100, "tokens_min": 1, "tokens_max": 1024}}
    assert workloads.shape("multidoc", catalog)["prefix_tokens"] == 0  # absent = no shared prefix


def test_distribution_needs_no_catalog():
    assert workloads.shape("distribution", None) is workloads.DISTRIBUTION_SHAPE


@pytest.mark.parametrize("name, msg", [("agentx-mvp", "replay"), ("nope", "no preset"),
                                       ("odd", "unknown keys"), ("short", "must state exactly")])
def test_refused(catalog, name, msg):
    with pytest.raises(ValueError, match=msg):
        workloads.shape(name, catalog)


BOUNDS = [  # (prompt override, message): each of the checks BLIS's validateDistributionParams makes
    ({"tokens_min": 0}, "must be >= 1"),
    ({"tokens_max": 0, "tokens_min": 0}, "must be >= 1"),
    ({"tokens_stdev": -1}, "must be >= 0"),
    ({"tokens_min": 900, "tokens": 900, "tokens_stdev": 900}, "must be <= tokens_max"),
    ({"tokens": 801}, r"must be in \[tokens_min, tokens_max\]"),
    ({"tokens": 1}, r"must be in \[tokens_min, tokens_max\]"),
    ({"tokens_stdev": 801}, r"must be in \[tokens_min, tokens_max\]"),
    ({"tokens_min": 50, "tokens_stdev": 10}, r"must be in \[tokens_min, tokens_max\]"),
    ({"tokens": 256.5}, "must be an integer"),
    ({"tokens_max": True}, "must be an integer"),
]


def _preset(prompt_over=None, prefix=0):
    prompt = {"tokens": 256, "tokens_stdev": 100, "tokens_min": 2, "tokens_max": 800, **(prompt_over or {})}
    out = {"tokens": 256, "tokens_stdev": 100, "tokens_min": 1, "tokens_max": 1024}
    return yaml.safe_dump({"prefix_tokens": prefix, "prompt": prompt, "output": out})


@pytest.mark.parametrize("over, msg", BOUNDS)
def test_bounds_refused(catalog, over, msg):
    (catalog / "workloads" / "bad.yaml").write_text(_preset(over))
    with pytest.raises(ValueError, match=msg):
        workloads.shape("bad", catalog)


def test_stdev_zero_is_allowed_below_min(catalog):
    (catalog / "workloads" / "fixed.yaml").write_text(_preset({"tokens_stdev": 0}))
    assert workloads.shape("fixed", catalog)["prompt"]["tokens_stdev"] == 0


@pytest.mark.parametrize("prefix", [-1, 1.5])
def test_prefix_refused(catalog, prefix):
    (catalog / "workloads" / "bad.yaml").write_text(_preset(prefix=prefix))
    with pytest.raises(ValueError, match="prefix_tokens"):
        workloads.shape("bad", catalog)


def test_bad_preset_refused_before_any_run(catalog, tmp_path):
    (catalog / "workloads" / "bad.yaml").write_text(_preset({"tokens_min": 900}))
    with pytest.raises(ValueError, match="tokens_min"):
        CachedRunner(BlisEnv("blis", catalog, "reg", tmp_path / "sc"),
                     Scenario("m", "h100", "bad", 8, "0.29.0"), tmp_path / "c")


def test_distribution_shape_passes_the_bounds():
    assert workloads._bounds_problem(workloads.DISTRIBUTION_SHAPE) is None


def test_cli_flags(catalog):
    f = workloads.cli_flags(workloads.shape("chatbot", catalog))
    assert f == ["--prefix-tokens", "0",
                 "--prompt-tokens", "256", "--prompt-tokens-stdev", "100",
                 "--prompt-tokens-min", "2", "--prompt-tokens-max", "800",
                 "--output-tokens", "256", "--output-tokens-stdev", "100",
                 "--output-tokens-min", "1", "--output-tokens-max", "1024"]


def test_run_one_passes_the_shape_not_the_name(catalog, tmp_path):
    rec = run_blis.run_one(FALSE, str(catalog), "reg", str(tmp_path / "sc"), "m", "h100", "0.29.0",
                           "chatbot", 80, 0, None, {"tp": 8, "num_instances": 1, "concurrency": 4})
    cmd = rec["cmd"]
    assert "--workload" not in cmd
    assert cmd[cmd.index("--prompt-tokens-max") + 1] == "800"
    assert rec["workload"]["name"] == "chatbot" and rec["workload"]["shape"]["output"]["tokens_max"] == 1024


def test_run_one_refuses_a_replay_workload_as_an_error_row(catalog, tmp_path):
    rec = run_blis.run_one(FALSE, str(catalog), "reg", str(tmp_path / "sc"), "m", "h100", "0.29.0",
                           "agentx-mvp", 80, 0, None, {"tp": 8, "num_instances": 1, "concurrency": 4})
    assert rec["returncode"] is None and rec["cmd"] is None and "replay" in rec["error"]


def test_cache_key_covers_the_shape(catalog, tmp_path):
    sc = Scenario("m", "h100", "chatbot", 8, "0.29.0")
    cfg = {"tp": 8, "num_instances": 1}
    legacy = cache_key(cfg, 8, 0, sc)
    chat = cache_key(cfg, 8, 0, sc, workloads.shape("chatbot", catalog))
    assert legacy != chat  # a pre-fix "chatbot" run drew 512/512 and must not be served as chatbot
    assert chat != cache_key(cfg, 8, 0, sc, workloads.DISTRIBUTION_SHAPE)
    env = BlisEnv("blis", catalog, "reg", tmp_path / "sc")
    assert CachedRunner(env, sc, tmp_path / "c").shape == workloads.shape("chatbot", catalog)
    with pytest.raises(ValueError, match="replay"):  # refused before any run
        CachedRunner(env, Scenario("m", "h100", "agentx-mvp", 8, "0.29.0"), tmp_path / "c")

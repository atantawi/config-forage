"""Re-measure canonical_config's no-op folds and the inert eplb fields on one BLIS stack.

Each fold claims a value runs byte-identically to its canonical form; each inert field claims BLIS
accepts it without pricing it. Both are measured here with the folds and the inert-field refusals
turned off, so every variant reaches BLIS as written (its own scenario file). One instance of
tp4*dp2 + EP at c=64, as the v0.14.0 measurements were. A control (a value BLIS does price) must
differ, or the comparison proves nothing. Prints a markdown table.

What is compared is BLIS's whole stdout, not the flat metrics a record keeps (a fold must leave every
field the same, not only the ones scoring reads): a v0.15.0 Result document parsed as JSON, less
DROPPED (fields two runs of one config, or two spellings of one deployment, differ in by
construction); an older stack's flat output as the list of every JSON object on it. A pair is
"identical" only when the rest is equal.

  BLIS_BIN=... BLIS_CATALOG=... BLIS_REGISTRY=... \\
    uv run python scripts/measure_folds.py v0.15.0 --model kimi-k2.5 --hardware h200
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import replace

from config_forage.blis import run_blis
from config_forage.blis.runner import BlisEnv, check_binary, requests_for
from config_forage.blis.stacks import STACKS

BASE = {"tp": 4, "dp": 2, "enable_expert_parallel": True, "num_instances": 1,
        "quantization": "fp8", "cache_dtype": "fp8", "concurrency": 64}
ON = {"enable_eplb": True}
# Result-document fields left out of the comparison: the run's own timing, the variant's scenario and
# deployment names (a hash of the deployment), and the echo of the engine block the variant states.
DROPPED = ("name", "runtime.wall_clock_s", "runtime.started_utc", "input.scenario.name",
           "input.deployment.name", "input.deployment.pools[*].engine")


def pairs(redundant: tuple[int, int], inert: bool = True):
    """(name, claim, variant, its canonical form); name is a FOLDS entry, "inert:<field>" or "control"
    (a value BLIS prices, which must move the metrics). `redundant`: the two num_redundant_experts
    counts EPLB-on variants use (each must fit in memory and keep the EP width dividing the experts)."""
    r0 = redundant[0]
    out = [
    ("long_prefill_token_threshold_zero", "lptt 0 = absent", {"long_prefill_token_threshold": 0}, {}),
    ("all2all_default", "allgather_reducescatter = absent", {"all2all_backend": "allgather_reducescatter"}, {}),
    ("all2all_alias", "pplx = allgather_reducescatter", {"all2all_backend": "pplx"},
     {"all2all_backend": "allgather_reducescatter"}),
    ("all2all_alias", "naive = allgather_reducescatter", {"all2all_backend": "naive"},
     {"all2all_backend": "allgather_reducescatter"}),
    ("eplb_disabled", "eplb off = absent", {"eplb": {"enable_eplb": False}}, {}),
    ("eplb_disabled", f"eplb off, {r0} redundant = absent",
     {"eplb": {"enable_eplb": False, "num_redundant_experts": r0}}, {}),
    ("eplb_zero_redundant", "eplb on, 0 redundant = eplb on",
     {"eplb": {**ON, "num_redundant_experts": 0}}, {"eplb": ON}),
    ]
    for f in ("window_size", "step_interval") if inert else ():
        for r in redundant:
            out.append((f"inert:{f}", f"eplb on, {r} redundant: {f} 1 = 1000000",
                        {"eplb": {**ON, "num_redundant_experts": r, f: 1}},
                        {"eplb": {**ON, "num_redundant_experts": r, f: 1000000}}))
    return out + [("control", "lptt 64 != absent", {"long_prefill_token_threshold": 64}, {}),
                  ("control", f"eplb on, {r0} redundant != eplb off",
                   {"eplb": {**ON, "num_redundant_experts": r0}}, {})]


def _drop(doc, path):
    head, _, rest = path.partition(".")
    key, star = (head[:-3], True) if head.endswith("[*]") else (head, False)
    if not isinstance(doc, dict) or key not in doc:
        return
    if not rest:
        del doc[key]
    elif star:
        for x in doc[key]:
            _drop(x, rest)
    else:
        _drop(doc[key], rest)


def comparable(stdout: str):
    """BLIS's stdout as compared here: the Result document less DROPPED, else every JSON object."""
    try:
        doc = json.loads(stdout)
    except ValueError:
        doc = None
    if isinstance(doc, dict) and doc.get("kind") == "Result":
        for f in DROPPED:
            _drop(doc, f)
        return doc
    return list(run_blis._iter_json_objects(stdout))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("stack", choices=sorted(STACKS))
    p.add_argument("--model", required=True)
    p.add_argument("--hardware", required=True)
    p.add_argument("--workload", default="chatbot")
    p.add_argument("--num-requests-base", type=int, default=1280)
    p.add_argument("--redundant", default="32,64", help="two num_redundant_experts counts (default 32,64)")
    p.add_argument("--no-inert", action="store_true", help="skip the inert eplb field pairs (for a stack "
                   "whose BLIS refuses window_size and step_interval itself, as v0.16.0 does)")
    p.add_argument("--set", default="{}", help="JSON merged into the base config, e.g. a model that needs "
                   'more memory for redundant experts: \'{"gpu_memory_utilization": 0.95}\'')
    a = p.parse_args(argv)
    base = {**BASE, **json.loads(a.set)}
    stack = STACKS[a.stack]
    ev = stack.engine_version          # a stack's tables hold at its own engine release only
    env = BlisEnv.from_env() if "BLIS_SCENARIOS" in os.environ else BlisEnv(
        *(os.environ[k] for k in ("BLIS_BIN", "BLIS_CATALOG", "BLIS_REGISTRY")), tempfile.mkdtemp(prefix="cf-folds-"))
    check_binary(env, stack)
    raw = replace(stack, measured_folds=frozenset(), inert_block_fields={})   # nothing folded or refused
    n = requests_for(base["concurrency"], a.num_requests_base)
    cache: dict[str, dict] = {}

    def observed(extra):
        k = repr(sorted(extra.items()))
        if k not in cache:
            out, real = {}, subprocess.run

            def capture(cmd, **kw):        # run_one keeps only a tail of stdout: keep all of it
                pr = real(cmd, **kw)
                out["stdout"] = pr.stdout
                return pr
            run_blis.subprocess.run = capture
            try:
                r = run_blis.run_one(str(env.blis), str(env.catalog), str(env.registry), str(env.scenarios_dir),
                                     a.model, a.hardware, ev, a.workload, n, 0, None, {**base, **extra}, stack=raw)
            finally:
                run_blis.subprocess.run = real
            if r["returncode"] != 0:
                sys.exit(f"{extra}: {r['error']}")
            cache[k] = comparable(out["stdout"])
        return cache[k]

    print(f"stack {stack.name}, engine {ev}, {a.model}/{a.hardware}/{a.workload}, 1 x tp4*dp2 + EP, "
          f"c=64, {n} requests, seed 0; base {base}\n")
    print(f"compared: BLIS stdout less {', '.join(DROPPED)}\n")
    print("| fold / field | claim | identical |\n|---|---|---|")
    verdict: dict[str, bool] = {}
    for name, claim, v, canon in pairs(tuple(int(x) for x in a.redundant.split(",")), not a.no_inert):
        same = observed(v) == observed(canon)
        print(f"| {name} | {claim} | {'yes' if same else 'no'} |")
        if name != "control":
            verdict[name] = verdict.get(name, True) and same
        elif same:
            sys.exit(f"control {claim!r} did not move the metrics: the comparison proves nothing")
    print("\nidentical on every pair: " + ", ".join(k for k, ok in verdict.items() if ok))
    print("not identical: " + (", ".join(k for k, ok in verdict.items() if not ok) or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

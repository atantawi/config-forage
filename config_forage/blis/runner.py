"""Cached BLIS runs: content-addressed JSON cache, thread-safe counters, thread-pool batches.

Budget unit is unique BLIS runs (`unique_runs`); cache hits are counted separately."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from config_forage import router
from config_forage.blis import run_blis, workloads
from config_forage.blis.stacks import DEFAULT_STACK, STACKS, Stack, stack_of
from config_forage.space import Scenario

T = TypeVar("T")
R = TypeVar("R")

# Back-compat alias; the runner's workload contract (the last pin) is explained in stacks.py.
# The last pin is the workload contract: runs before it passed --workload, which closed-loop BLIS
# ignored (they all drew from workloads.DISTRIBUTION_SHAPE), so cf-bench and cf-grid refuse to resume into them.
PINS = STACKS[DEFAULT_STACK].pins


# Engine keys accepted from contract 2 on. A config using one was refused (an error row, cached)
# before; keying it under contract 2 keeps it off those rows without rotating PINS, so every other
# config keeps its key and its cached run.
ENGINE_KEYS_V2 = ("all2all_backend", "long_prefill_token_threshold", "eplb")


def engine_contract(space) -> dict:
    """The engine-key contract a space's stored results (cf-grid sidecar, cf-bench manifest) carry:
    {"engine_keys": 2} when a knob names one of ENGINE_KEYS_V2 (a pinned engine setting is a
    single-valued knob, so it counts), else {} -- so a space without them writes exactly the metadata
    it did before and its existing results resume; one with them refuses results made before."""
    return {"engine_keys": 2} if any(n in ENGINE_KEYS_V2 for n in space.names) else {}


def requests_for(concurrency: int, base: int = 80) -> int:
    """Requests per probe point: max(base, 4c) (as in inference-sim/blis-config-search run_search.py)."""
    return max(base, concurrency * 4)


@dataclass
class BlisEnv:
    blis: Path
    catalog: Path
    registry: Path
    scenarios_dir: Path
    timeout: int = 900  # a timeout is not cached but fails the probe point: keep it far above the slowest run

    @classmethod
    def from_env(cls) -> BlisEnv:
        e = os.environ
        return cls(Path(e["BLIS_BIN"]), Path(e["BLIS_CATALOG"]), Path(e["BLIS_REGISTRY"]),
                   Path(e["BLIS_SCENARIOS"]))


def _git(d: Path, *args: str) -> str | None:
    """`git -C d args` stdout stripped, or None when it fails (not a checkout, no exact tag)."""
    try:
        r = subprocess.run(["git", "-C", str(d), *args], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def check_binary(env: BlisEnv, stack: Stack) -> None:
    """Refuse a BLIS stack other than `stack`, before any run is keyed under the stack's pins
    (ValueError): when the binary sits in an inference-sim checkout (main.go beside it), its HEAD must
    be stack.sim_sha; a binary outside a checkout (e.g. a copied or cross-built one) cannot be
    checked, so it gets a warning on stderr instead. Either way the catalog and registry checkouts
    must be exactly at stack.catalog_tag and stack.registry_tag."""
    sim = Path(env.blis).parent
    if not (sim / "main.go").exists():
        print(f"config-forage: warning: {env.blis} is not inside an inference-sim checkout; cannot check "
              f"it is stack {stack.name} ({stack.sim_sha})", file=sys.stderr)
    else:
        head = _git(sim, "rev-parse", "HEAD")
        if head != stack.sim_sha:
            raise ValueError(f"{env.blis} is built from inference-sim {head}, not stack {stack.name}'s "
                             f"{stack.sim_sha}; point BLIS_BIN at that stack's binary")
    for what, d, tag in (("catalog", env.catalog, stack.catalog_tag), ("registry", env.registry, stack.registry_tag)):
        have = _git(Path(d), "describe", "--tags", "--exact-match")
        if have != tag:
            raise ValueError(f"{what} {d} is at {have or 'no exact tag'}, not stack {stack.name}'s {tag}; "
                             f"point BLIS_{what.upper()} at that stack's checkout")


def cache_key(config: dict, concurrency: int, seed: int, scenario: Scenario, shape: dict | None = None) -> str:
    """`shape` is the workload's token distribution (workloads.shape), keyed so a run is cached under
    what it drew from. Keys written before it existed (no "shape") are for runs that all drew from
    workloads.DISTRIBUTION_SHAPE whatever their workload name said, and never match a key with one.
    Likewise a config using ENGINE_KEYS_V2 is keyed under that contract (`engine_keys`); callers pass
    the canonical config (run_blis.canonical_config), so a no-op spelling of one of them is not."""
    payload = {"config": config, "concurrency": concurrency, "seed": seed, "workload": scenario.workload,
               "num_requests": requests_for(concurrency, scenario.num_requests_base),
               "model": scenario.model, "hardware": scenario.hardware,
               "engine_version": scenario.engine_version, "pins": stack_of(scenario).pins}
    if shape is not None:
        payload["shape"] = shape
    if any(k in config for k in ENGINE_KEYS_V2):
        payload["engine_keys"] = 2
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class CachedRunner:
    def __init__(self, env: BlisEnv | None, scenario: Scenario, cache_dir: Path, run_fn: Callable | None = None):
        """The scenario's engine_version must be its stack's (ValueError). With an env, the binary and checkouts are checked against the scenario's stack
        (check_binary) and the workload is resolved to its shape here, so an unknown or replay
        workload is refused before any run. Without one (a test double), the shape is not keyed."""
        self.env, self.scenario, self.cache_dir = env, scenario, Path(cache_dir)
        self.stack = stack_of(scenario)
        if scenario.engine_version != self.stack.engine_version:
            raise ValueError(f"scenario engine_version {scenario.engine_version!r} is not stack "
                             f"{self.stack.name}'s {self.stack.engine_version!r}; a stack runs one engine release")
        if env is not None:
            check_binary(env, self.stack)
        self.shape = workloads.shape(scenario.workload, env.catalog) if env is not None else None
        self.run_fn = run_fn or self._blis_run
        self.unique_runs = 0
        self.cache_hits = 0
        self.timeouts = 0  # BLIS runs that hit the timeout (returncode -1; never cached)
        self._lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}

    def _blis_run(self, config: dict, concurrency: int, seed: int) -> dict:
        s, e = self.scenario, self.env
        return run_blis.run_one(e.blis, e.catalog, e.registry, e.scenarios_dir, s.model, s.hardware,
                                s.engine_version, s.workload, requests_for(concurrency, s.num_requests_base),
                                seed, None, {**config, "concurrency": concurrency}, timeout=e.timeout,
                                stack=self.stack)

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _load(self, key: str) -> dict | None:
        try:
            return json.loads(self._path(key).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _shed_guard(self, rec: dict, concurrency: int) -> dict:
        """A run that admitted fewer requests than offered is an error: BLIS leaves admission-rejected
        requests out of injected_requests, so its throughput/latency describe only the admitted load.
        Derived on return (hit or miss), so cached records written before this guard stay valid."""
        if rec.get("error") is not None:
            return rec
        inj = (rec.get("metrics") or {}).get("injected_requests")
        offered = requests_for(concurrency, self.scenario.num_requests_base)
        if inj is not None and inj < offered:
            return {**rec, "error": f"shed: injected={inj} < offered={offered}"}
        return rec

    def run(self, config: dict, concurrency: int, seed: int | None = None, shed_guard: bool = True) -> dict:
        """Router knobs are translated first (router.to_blis), so the cache key covers the
        EndpointPickerConfig BLIS actually runs, not just the knob values. A config they cannot
        translate is passed on as is: run_blis refuses it and records an error row.
        shed_guard=False (ladder scoring) returns the record without _shed_guard; the cached
        record is the same either way, as the guard is applied only on return."""
        guard = self._shed_guard if shed_guard else (lambda rec, _c: rec)
        seed = self.scenario.seed if seed is None else seed
        families = self.scenario.router_families
        if families and config.get(router.FAMILY_KNOB) in families:
            config = router.to_blis(config, families)  # a file family's errors are the space's: raise
        else:
            try:
                config = router.to_blis(config)
            except ValueError:
                pass
        # measured no-op engine values (e.g. eplb with enable_eplb false) share the default's run
        config = run_blis.canonical_config(config, self.scenario.engine_version, self.stack)
        key = cache_key(config, concurrency, seed, self.scenario, self.shape)
        with self._lock:
            klock = self._key_locks.setdefault(key, threading.Lock())
        with klock:  # concurrent requests for one key call BLIS once
            rec = self._load(key)
            if rec is not None:
                with self._lock:
                    self.cache_hits += 1
                return {**guard(rec, concurrency), "cache_key": key}
            rec = self.run_fn(config, concurrency, seed)
            with self._lock:
                self.unique_runs += 1
                self.timeouts += rec.get("returncode") == -1
            if rec.get("returncode") != -1:  # BLIS is deterministic; only a timeout is not cached
                p = self._path(key)
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
                tmp.write_text(json.dumps(rec, default=str))  # cmd holds Paths
                os.replace(tmp, p)
            return {**guard(rec, concurrency), "cache_key": key}


def run_batch(fn: Callable[[T], R], items: list[T], workers: int | None = None) -> list[R]:
    """Map fn over items on threads (BLIS is a subprocess), preserving order."""
    if not items:
        return []
    n = workers or max(1, min((os.cpu_count() or 2) - 1, 32))
    with ThreadPoolExecutor(max_workers=min(n, len(items))) as ex:
        return list(ex.map(fn, items))

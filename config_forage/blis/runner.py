"""Cached BLIS runs: content-addressed JSON cache, thread-safe counters, thread-pool batches.

Budget unit is unique BLIS runs (`unique_runs`); cache hits are counted separately."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from config_forage import router
from config_forage.blis import run_blis
from config_forage.space import Scenario

T = TypeVar("T")
R = TypeVar("R")

PINS = ("inference-sim v0.14.0 f2ba6bb1c189ed5a56933cfa3d0927eae3836ee9", "kernel v0.4.0",
        "blis-vllm v0.32.0", "catalog 0.3.0", "registry v0.2.0")


def requests_for(concurrency: int, base: int = 80) -> int:
    """Requests per probe point: max(base, 4c) (as in inference-sim/blis-config-search run_search.py)."""
    return max(base, concurrency * 4)


@dataclass
class BlisEnv:
    blis: Path
    catalog: Path
    registry: Path
    scenarios_dir: Path
    timeout: int = 120

    @classmethod
    def from_env(cls) -> BlisEnv:
        e = os.environ
        return cls(Path(e["BLIS_BIN"]), Path(e["BLIS_CATALOG"]), Path(e["BLIS_REGISTRY"]),
                   Path(e["BLIS_SCENARIOS"]))


def cache_key(config: dict, concurrency: int, seed: int, scenario: Scenario) -> str:
    payload = {"config": config, "concurrency": concurrency, "seed": seed, "workload": scenario.workload,
               "num_requests": requests_for(concurrency, scenario.num_requests_base),
               "model": scenario.model, "hardware": scenario.hardware,
               "engine_version": scenario.engine_version, "pins": PINS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class CachedRunner:
    def __init__(self, env: BlisEnv | None, scenario: Scenario, cache_dir: Path, run_fn: Callable | None = None):
        self.env, self.scenario, self.cache_dir = env, scenario, Path(cache_dir)
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
                                seed, None, {**config, "concurrency": concurrency}, timeout=e.timeout)

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

    def run(self, config: dict, concurrency: int, seed: int | None = None) -> dict:
        """Router knobs are translated first (router.to_blis), so the cache key covers the
        EndpointPickerConfig BLIS actually runs, not just the knob values."""
        seed = self.scenario.seed if seed is None else seed
        config = router.to_blis(config)
        key = cache_key(config, concurrency, seed, self.scenario)
        with self._lock:
            klock = self._key_locks.setdefault(key, threading.Lock())
        with klock:  # concurrent requests for one key call BLIS once
            rec = self._load(key)
            if rec is not None:
                with self._lock:
                    self.cache_hits += 1
                return {**self._shed_guard(rec, concurrency), "cache_key": key}
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
            return {**self._shed_guard(rec, concurrency), "cache_key": key}


def run_batch(fn: Callable[[T], R], items: list[T], workers: int | None = None) -> list[R]:
    """Map fn over items on threads (BLIS is a subprocess), preserving order."""
    if not items:
        return []
    n = workers or max(1, min((os.cpu_count() or 2) - 1, 32))
    with ThreadPoolExecutor(max_workers=min(n, len(items))) as ex:
        return list(ex.map(fn, items))

"""scripts/bsa_sweep.py refuses a grid that is not its space's (no BLIS needed)."""

import importlib.util
import json
from pathlib import Path

import pytest

from config_forage.blis.runner import PINS
from config_forage.space import load_space, space_hash

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("bsa_sweep", ROOT / "scripts" / "bsa_sweep.py")
bsa_sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bsa_sweep)
SPACE = load_space(ROOT / "tests" / "data" / "tiny-space.yaml")


def _grid(tmp_path, meta):
    g = tmp_path / "grid.jsonl"
    g.write_text("")
    if meta is not None:
        (tmp_path / "grid.jsonl.meta.json").write_text(json.dumps(meta))
    return g


def test_matching_grid_is_accepted(tmp_path):
    bsa_sweep.check_grid(SPACE, _grid(tmp_path, {"space_hash": space_hash(SPACE), "pins": list(PINS)}))


@pytest.mark.parametrize("meta, msg", [
    (None, "no sidecar"),
    ({"space_hash": "other", "pins": list(PINS)}, "space_hash"),
    ({"space_hash": space_hash(SPACE), "pins": ["old stack"]}, "pins"),
])
def test_wrong_or_unknown_grid_is_refused(tmp_path, meta, msg):
    with pytest.raises(ValueError, match=msg):
        bsa_sweep.check_grid(SPACE, _grid(tmp_path, meta))

"""scripts/bsa_sweep.py refuses a grid that is not its space's (no BLIS needed)."""

import importlib.util
import json
from pathlib import Path

import pytest

from config_forage.blis.runner import PINS
from config_forage.feasibility import enumerate_feasible
from config_forage.space import load_space, space_hash

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("bsa_sweep", ROOT / "scripts" / "bsa_sweep.py")
bsa_sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bsa_sweep)
SPACE = load_space(ROOT / "tests" / "data" / "tiny-space.yaml")


GOOD = {"space_hash": space_hash(SPACE), "pins": list(PINS)}
ALL = [list(d) for d in enumerate_feasible(SPACE)]


def _grid(tmp_path, meta, decisions=ALL):
    g = tmp_path / "grid.jsonl"
    g.write_text("".join(json.dumps({"decision": d, "score": -1.0}) + "\n" for d in decisions))
    if meta is not None:
        (tmp_path / "grid.jsonl.meta.json").write_text(json.dumps(meta))
    return g


def test_matching_grid_is_accepted(tmp_path):
    assert len(bsa_sweep.check_grid(SPACE, _grid(tmp_path, GOOD))) == len(ALL)


@pytest.mark.parametrize("meta, msg", [
    (None, "no sidecar"),
    ({"space_hash": "other", "pins": list(PINS)}, "space_hash"),
    ({"space_hash": space_hash(SPACE), "pins": ["old stack"]}, "pins"),
])
def test_wrong_or_unknown_grid_is_refused(tmp_path, meta, msg):
    with pytest.raises(ValueError, match=msg):
        bsa_sweep.check_grid(SPACE, _grid(tmp_path, meta))


def test_interrupted_grid_is_refused(tmp_path):
    """cf-grid writes the sidecar before probing, so a killed grid has valid metadata but few rows."""
    with pytest.raises(ValueError, match=f"{len(ALL) - 1} of the space's {len(ALL)} feasible decisions missing"):
        bsa_sweep.check_grid(SPACE, _grid(tmp_path, GOOD, ALL[:1]))


def test_truncated_last_line_counts_as_missing(tmp_path):
    g = _grid(tmp_path, GOOD)
    g.write_text(g.read_text()[:-5])  # the last row cut short by a kill
    with pytest.raises(ValueError, match="1 of the space"):
        bsa_sweep.check_grid(SPACE, g)

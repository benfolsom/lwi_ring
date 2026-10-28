"""Configuration contract (no physics runs): canonical defaults kept, new options inert by default, unified aliases, conflicts rejected."""
from __future__ import annotations

import dataclasses
import json

import pytest

from cases import DATA, J1


def test_canonical_defaults_kept():
    """Every field of the canonical ring_driver Config exists with the same default (data/canonical_defaults.json)."""
    from lwi_ring.driver import Config
    canonical = json.loads((DATA / "canonical_defaults.json").read_text())
    ours = {f.name: f.default for f in dataclasses.fields(Config)}
    for name, default in canonical.items():
        assert name in ours, name
        assert json.loads(json.dumps(ours[name])) == default, name


def test_defaults_validate():
    from lwi_ring.driver import Config
    Config().validate()


def test_aliases():
    from lwi_ring.driver import Config
    base = {k: v for k, v in J1.items()}
    c = Config(**dict(base, column_recentre_stations=0, column_drift_resets=4))
    c.validate()
    assert c.electron_resets() == 4
    c = Config(**dict(base, column_recentre_delay_steps=2))
    c.validate()
    assert c.station_delay_steps() == 2
    with pytest.raises(ValueError, match="same option"):
        Config(**dict(base, column_recentre_stations=0, dyn_resets=2, column_drift_resets=4)).validate()
    with pytest.raises(ValueError, match="same option"):
        Config(**dict(base, rem_delay_steps=1, column_recentre_delay_steps=2)).validate()
    with pytest.raises(ValueError, match="rem_station_pos"):
        Config(**dict(base, column_drift_resets=2)).validate()       # stations with resets need positions, as for dyn_resets


@pytest.mark.parametrize("bad", [
    dict(column_drift_omega_overlap=True),                                   # esize option on dynamic columns
    dict(rem_birth_ratio=1.05),                                              # remedy option without the fixed gun size
    dict(column_electrons="rigid", dyn_resets=2),                            # dynamic option on rigid columns
    dict(wfb_enable=True, column_electrons="rigid"),
])
def test_conflicts_rejected(bad):
    from lwi_ring.driver import Config
    with pytest.raises(ValueError):
        Config(**dict(J1, **bad)).validate()

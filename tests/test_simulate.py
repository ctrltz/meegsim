import numpy as np
import pytest

from mock import patch, Mock

from meegsim.source_groups import PointSourceGroup
from meegsim.simulate import SourceSimulator

from utils.prepare import prepare_source_space


def test_sourcesimulator_add_point_sources():
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )
    sim = SourceSimulator(src)

    # Add one group with auto-generated names
    sim.add_point_sources(
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        np.ones((4, 100))
    )

    assert len(sim._source_groups) == 1, \
        f"Expected one source group to be created, got {len(sim._source_groups)}"
    assert len(sim._noise_groups) == 0, \
        f"Expected no noise groups to be created, got {len(sim._noise_groups)}"
    assert len(sim._sources) == 4, \
        f"Expected four sources to be created, got {len(sim._sources)}"
    
    # Add one group with custom names
    custom_names = ['s1', 's2', 's3', 's4']
    sim.add_point_sources(
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        np.ones((4, 100)),
        names=custom_names
    )

    assert len(sim._source_groups) == 2, \
        f"Expected two source groups to be created, got {len(sim._source_groups)}"
    assert len(sim._noise_groups) == 0, \
        f"Expected no noise groups to be created, got {len(sim._noise_groups)}"
    assert len(sim._sources) == 8, \
        f"Expected eight sources to be created, got {len(sim._sources)}"
    assert all([name in sim._sources for name in custom_names]), \
        f"Provided source names were not used properly"
    
    # Add one group with already existing names
    with pytest.raises(ValueError):
        sim.add_point_sources(
            [(0, 0), (1, 1)],
            np.ones((2, 100)),
            names=['s1', 's4']
        )


def test_sourcesimulator_add_noise_sources():
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )
    sim = SourceSimulator(src)

    # Add one group (names are always auto-generated)
    sim.add_noise_sources(
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        np.ones((4, 100))
    )

    assert len(sim._noise_groups) == 1, \
        f"Expected one noise group to be created, got {len(sim._noise_groups)}"
    assert len(sim._source_groups) == 0, \
        f"Expected no source groups to be created, got {len(sim._source_groups)}"
    assert len(sim._sources) == 4, \
        f"Expected four sources to be created, got {len(sim._sources)}"
    
    # Add second group (with auto-generated names)
    sim.add_noise_sources(
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        np.ones((4, 100))
    )

    assert len(sim._noise_groups) == 2, \
        f"Expected two noise groups to be created, got {len(sim._noise_groups)}"
    assert len(sim._source_groups) == 0, \
        f"Expected no source groups to be created, got {len(sim._source_groups)}"
    assert len(sim._sources) == 8, \
        f"Expected eight sources to be created, got {len(sim._sources)}"

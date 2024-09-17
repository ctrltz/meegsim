import meegsim
import numpy as np
import pytest

from collections import namedtuple
from mock import patch, Mock

from meegsim.simulate import SourceSimulator, _simulate
from meegsim.source_groups import PointSourceGroup

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


def test_sourcesimulator_simulate_empty_raises():
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )
    sim = SourceSimulator(src)

    with pytest.raises(ValueError, match='No sources were added'):
        sim.simulate(sfreq=250, duration=30, random_state=0)


@patch('meegsim.simulate._simulate', return_value=0)
def test_sourcesimulator_simulate(simulate_mock):
    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )
    sim = SourceSimulator(src)
    sim.add_point_sources([(0, 0)], np.ones((1, 100)))
    sim.simulate(sfreq=250, duration=30)

    # Check that the _simulate was called correctly
    simulate_mock.assert_called()
    assert simulate_mock.call_args.kwargs['random_state'] is None

    sim.simulate(sfreq=250, duration=30, random_state=0)

    # Check that the _simulate was called correctly
    simulate_mock.assert_called()
    assert simulate_mock.call_args.kwargs['random_state'] == 0


def test_simulate():
    # return mock PointSource's
    # noise sources are created first (1 + 3), then actual sources (2)
    MockSource = namedtuple("MockSource", ['name'])
    simulate_mock = Mock(side_effect=[
        [MockSource(name='s1')],
        [MockSource(name='s4'), MockSource(name='s5'), MockSource(name='s6')],
        [MockSource(name='s2'), MockSource(name='s3')],
    ])

    src = prepare_source_space(
        types=['surf', 'surf'],
        vertices=[[0, 1], [0, 1]]
    )

    # some dummy data - 2 sources + (1 + 3 = 4) noise sources expected
    source_groups = [
        PointSourceGroup(2, [(0, 0), (0, 1)], 
                         np.ones((2, 100)), None, dict(), []),
    ]
    noise_groups = [
        PointSourceGroup(1, [(0, 0)], np.array([0]), None, dict(), []),
        PointSourceGroup(3, [(0, 0), (0, 1), (1, 0)], 
                         np.ones((3, 100)), None, dict(), []),
    ]

    with patch.object(meegsim.source_groups.PointSourceGroup,
                      'simulate', simulate_mock):
        sc = _simulate(source_groups, noise_groups, src, 
                       sfreq=250, duration=30, random_state=0)
        
        assert len(simulate_mock.call_args_list) == 3, \
            f"Expected three calls of PointSourceGroup.simulate method"

        random_states = [kall.kwargs['random_state'] == 0
                         for kall in simulate_mock.call_args_list]
        assert all(random_states), "random_state was not passed correctly"

        assert len(sc._sources) == 2, f"Expected 2 sources, got {len(sc._sources)}"
        assert len(sc._noise_sources) == 4, \
            f"Expected 4 sources, got {len(sc._noise_sources)}"

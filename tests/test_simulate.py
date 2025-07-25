import meegsim
import networkx as nx
import numpy as np
import pytest

from mock import patch, Mock

from meegsim.simulate import SourceSimulator, _simulate
from meegsim.source_groups import PointSourceGroup, PatchSourceGroup

from utils.prepare import prepare_source_space, prepare_forward, prepare_point_source


def test_sourcesimulator_add_point_sources():
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    sim = SourceSimulator(src)

    # Add one group with auto-generated names
    sim.add_point_sources([(0, 0), (0, 1), (1, 0), (1, 1)], np.ones((4, 100)))

    assert (
        len(sim._source_groups) == 1
    ), f"Expected one source group to be created, got {len(sim._source_groups)}"
    assert (
        len(sim._noise_groups) == 0
    ), f"Expected no noise groups to be created, got {len(sim._noise_groups)}"
    assert (
        len(sim._sources) == 4
    ), f"Expected four sources to be created, got {len(sim._sources)}"

    # Add one group with custom names
    custom_names = ["s1", "s2", "s3", "s4"]
    sim.add_point_sources(
        [(0, 0), (0, 1), (1, 0), (1, 1)], np.ones((4, 100)), names=custom_names
    )

    assert (
        len(sim._source_groups) == 2
    ), f"Expected two source groups to be created, got {len(sim._source_groups)}"
    assert (
        len(sim._noise_groups) == 0
    ), f"Expected no noise groups to be created, got {len(sim._noise_groups)}"
    assert (
        len(sim._sources) == 8
    ), f"Expected eight sources to be created, got {len(sim._sources)}"
    assert all(
        [name in sim._sources for name in custom_names]
    ), "Provided source names were not used properly"

    # Add one group with already existing names
    with pytest.raises(ValueError):
        sim.add_point_sources([(0, 0), (1, 1)], np.ones((2, 100)), names=["s1", "s4"])


def test_sourcesimulator_add_patch_sources():
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    sim = SourceSimulator(src)

    # Add one group with auto-generated names
    sim.add_patch_sources([(0, [0, 1]), (1, [0, 1])], np.ones((2, 100)))

    assert (
        len(sim._source_groups) == 1
    ), f"Expected one source group to be created, got {len(sim._source_groups)}"
    assert (
        len(sim._noise_groups) == 0
    ), f"Expected no noise groups to be created, got {len(sim._noise_groups)}"
    assert (
        len(sim._sources) == 2
    ), f"Expected two sources to be created, got {len(sim._sources)}"

    # Add one group with custom names
    custom_names = ["s1", "s2"]
    sim.add_patch_sources(
        [(0, [0, 1]), (1, [0, 1])], np.ones((2, 100)), names=custom_names
    )

    assert (
        len(sim._source_groups) == 2
    ), f"Expected two source groups to be created, got {len(sim._source_groups)}"
    assert (
        len(sim._noise_groups) == 0
    ), f"Expected no noise groups to be created, got {len(sim._noise_groups)}"
    assert (
        len(sim._sources) == 4
    ), f"Expected four sources to be created, got {len(sim._sources)}"
    assert all(
        [name in sim._sources for name in custom_names]
    ), "Provided source names were not used properly"

    # Add one group with already existing names
    with pytest.raises(ValueError):
        sim.add_patch_sources(
            [(0, [0, 1]), (1, [0, 1])], np.ones((2, 100)), names=["s1", "s4"]
        )


def test_sourcesimulator_add_noise_sources():
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    sim = SourceSimulator(src)

    # Add one group (names are always auto-generated)
    sim.add_noise_sources([(0, 0), (0, 1), (1, 0), (1, 1)], np.ones((4, 100)))

    assert (
        len(sim._noise_groups) == 1
    ), f"Expected one noise group to be created, got {len(sim._noise_groups)}"
    assert (
        len(sim._source_groups) == 0
    ), f"Expected no source groups to be created, got {len(sim._source_groups)}"
    assert (
        len(sim._sources) == 4
    ), f"Expected four sources to be created, got {len(sim._sources)}"

    # Add second group (with auto-generated names)
    sim.add_noise_sources([(0, 0), (0, 1), (1, 0), (1, 1)], np.ones((4, 100)))

    assert (
        len(sim._noise_groups) == 2
    ), f"Expected two noise groups to be created, got {len(sim._noise_groups)}"
    assert (
        len(sim._source_groups) == 0
    ), f"Expected no source groups to be created, got {len(sim._source_groups)}"
    assert (
        len(sim._sources) == 8
    ), f"Expected eight sources to be created, got {len(sim._sources)}"


@patch("meegsim.simulate.check_coupling", return_value={"param": 1})
def test_sourcesimulator_set_coupling_tuple(check_coupling_mock):
    from meegsim.coupling import ppc_von_mises

    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    sim = SourceSimulator(src)

    sim.add_point_sources(
        location=[(0, 0), (1, 1)], waveform=np.ones((2, 100)), names=["s1", "s2"]
    )

    sim.set_coupling(
        ("s1", "s2"), kappa=1, phase_lag=0, method=ppc_von_mises, fmin=8, fmax=12
    )

    # Check that the mock function was called
    check_coupling_mock.assert_called()

    # Check that the coupling graph is updated correctly
    assert sim._coupling_graph.has_edge("s1", "s2")

    edge_data = sim._coupling_graph.get_edge_data("s1", "s2")
    assert edge_data["param"] == 1  # mock output is saved


@patch("meegsim.simulate.check_coupling", return_value={"param": 1})
def test_sourcesimulator_set_coupling_dict(check_coupling_mock):
    from meegsim.coupling import ppc_von_mises

    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    sim = SourceSimulator(src)

    sim.add_point_sources(
        location=[(0, 0), (1, 1)], waveform=np.ones((2, 100)), names=["s1", "s2"]
    )

    sim.set_coupling(
        {
            ("s1", "s2"): dict(kappa=1, phase_lag=0),
        },
        method=ppc_von_mises,
        fmin=8,
        fmax=12,
    )

    # Check that the mock function was called
    check_coupling_mock.assert_called()

    # Check that the coupling graph is updated correctly
    assert sim._coupling_graph.has_edge("s1", "s2")

    edge_data = sim._coupling_graph.get_edge_data("s1", "s2")
    assert edge_data["param"] == 1  # mock output is saved


def test_sourcesimulator_is_local_snr_adjusted_false():
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    sim = SourceSimulator(src, snr_mode="local")

    # Add noise sources
    sim.add_noise_sources([(0, 0), (0, 1), (1, 0), (1, 1)], np.ones((4, 100)))
    assert not sim.is_local_snr_adjusted

    # Add point sources WITHOUT adjustment of SNR
    sim.add_point_sources([(0, 0), (0, 1), (1, 0), (1, 1)], np.ones((4, 100)))
    assert not sim.is_local_snr_adjusted

    # Add patch sources WITHOUT adjustment of SNR
    sim.add_patch_sources([(0, [0, 1]), (1, [0, 1])], np.ones((2, 100)))
    assert not sim.is_local_snr_adjusted


def test_sourcesimulator_is_local_snr_adjusted_ignored():
    # In global SNR mode, source-specific SNR should be ignored
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    sim = SourceSimulator(src)

    # Add point sources WITH adjustment of SNR
    sim.add_point_sources(
        [(0, [0, 1]), (1, [0, 1])],
        np.ones((2, 100)),
        snr=5,
        snr_params=dict(fmin=8, fmax=12),
    )
    assert not sim.is_local_snr_adjusted

    # Add patch sources WITH adjustment of SNR
    sim.add_patch_sources(
        [(0, [0, 1]), (1, [0, 1])],
        np.ones((2, 100)),
        snr=5,
        snr_params=dict(fmin=8, fmax=12),
    )
    assert not sim.is_local_snr_adjusted


def test_sourcesimulator_is_snr_adjusted_true_point_source():
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    sim = SourceSimulator(src, snr_mode="local")

    # Add point sources WITH adjustment of SNR
    sim.add_point_sources(
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        np.ones((4, 100)),
        snr=10,
        snr_params=dict(fmin=8, fmax=12),
    )

    # SNR should be adjusted now
    assert sim.is_local_snr_adjusted

    # Forward model is required for simulations
    with pytest.raises(ValueError, match="A forward model"):
        sim.simulate(sfreq=100, duration=30)


def test_sourcesimulator_is_snr_adjusted_true_patch_source():
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    sim = SourceSimulator(src, snr_mode="local")

    # Add point sources WITH adjustment of SNR
    sim.add_patch_sources(
        [(0, [0, 1]), (1, [0, 1])],
        np.ones((2, 100)),
        snr=10,
        snr_params=dict(fmin=8, fmax=12),
    )

    # SNR should be adjusted now
    assert sim.is_local_snr_adjusted

    # Forward model is required for simulations
    with pytest.raises(ValueError, match="A forward model"):
        sim.simulate(sfreq=100, duration=30)


def test_sourcesimulator_simulate_empty_raises():
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    sim = SourceSimulator(src)

    with pytest.raises(ValueError, match="No sources were added"):
        sim.simulate(sfreq=250, duration=30, random_state=0)


@patch("meegsim.simulate._simulate", return_value=([], []))
def test_sourcesimulator_simulate(simulate_mock):
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    sim = SourceSimulator(src)
    sim.add_point_sources([(0, 0)], np.ones((1, 100)))
    sim.simulate(sfreq=250, duration=30)

    # Check that the _simulate was called correctly
    simulate_mock.assert_called()
    assert simulate_mock.call_args.kwargs["random_state"] is None

    sim.simulate(sfreq=250, duration=30, random_state=0)

    # Check that the _simulate was called correctly
    simulate_mock.assert_called()
    assert simulate_mock.call_args.kwargs["random_state"] == 0


def test_simulate():
    # return mock PointSource's
    # noise sources are created first (1 + 3), then actual sources (2)
    simulate_mock = Mock(
        side_effect=[
            [prepare_point_source(name="s1")],
            [
                prepare_point_source(name="s4"),
                prepare_point_source(name="s5"),
                prepare_point_source(name="s6"),
            ],
            [prepare_point_source(name="s2"), prepare_point_source(name="s3")],
        ]
    )

    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])

    # some dummy data - 2 sources + (1 + 3 = 4) noise sources expected
    source_groups = [
        PointSourceGroup(
            2, [(0, 0), (0, 1)], np.ones((2, 100)), None, dict(), 1, ["s2", "s3"]
        ),
    ]
    noise_groups = [
        PointSourceGroup(1, [(0, 0)], np.array([0]), None, dict(), 1, ["s1"]),
        PointSourceGroup(
            3,
            [(0, 0), (0, 1), (1, 0)],
            np.ones((3, 100)),
            None,
            dict(),
            1,
            ["s4", "s5", "s6"],
        ),
    ]

    with patch.object(
        meegsim.source_groups.PointSourceGroup, "simulate", simulate_mock
    ):
        sfreq = 100
        duration = 5
        times = np.arange(0, sfreq * duration) / sfreq
        sources, noise_sources = _simulate(
            source_groups=source_groups,
            noise_groups=noise_groups,
            coupling_graph=nx.Graph(),
            snr_mode="global",
            snr_global=None,
            snr_params=dict(),
            is_local_snr_adjusted=False,
            src=src,
            times=times,
            fwd=None,
            base_std=1e-9,
            random_state=0,
        )

        assert (
            len(simulate_mock.call_args_list) == 3
        ), "Expected three calls of PointSourceGroup.simulate method"

        assert len(sources) == 2, f"Expected 2 sources, got {len(sources)}"
        assert len(noise_sources) == 4, f"Expected 4 sources, got {len(noise_sources)}"

        # Check that all source waveform were scaled by the base std
        for s in sources.values():
            assert np.allclose(s.waveform, 1e-9)
        for s in noise_sources.values():
            assert np.allclose(s.waveform, 1e-9)


def test_simulate_std_adjustment():
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    fwd = prepare_forward(5, 4)

    # Define source groups
    source_groups = [
        PointSourceGroup(
            n_sources=1,
            location=[(0, 0)],
            waveform=np.ones((1, 100)),
            snr=None,
            snr_params=dict(),
            std=[2],
            names=["point"],
        ),
        PatchSourceGroup(
            n_sources=1,
            location=[(1, [0, 1])],
            # adjust the waveform to account for scaling in .data
            waveform=np.ones((1, 100)) * np.sqrt(2),
            snr=None,
            snr_params=dict(),
            std=[3],
            extents=[None],
            subject=None,
            subjects_dir=None,
            names=["patch"],
        ),
    ]
    noise_groups = [
        PointSourceGroup(
            n_sources=2,
            location=[(0, 1), (1, 0)],
            waveform=np.ones((2, 100)),
            snr=None,
            snr_params=dict(),
            std=[0.5, 1],
            names=["noise1", "noise2"],
        ),
    ]

    sfreq = 20
    duration = 5
    times = np.arange(0, sfreq * duration) / sfreq
    sources, noise_sources = _simulate(
        source_groups=source_groups,
        noise_groups=noise_groups,
        coupling_graph=nx.Graph(),
        snr_mode="local",
        snr_global=None,
        snr_params=dict(),
        is_local_snr_adjusted=False,
        src=src,
        times=times,
        fwd=fwd,
        base_std=1,
        random_state=0,
    )

    # Check that all waveforms were scaled according to the requested std
    assert np.allclose(sources["point"].data, 2.0), "point"
    assert np.allclose(sources["patch"].data, 3.0), "patch"
    assert np.allclose(noise_sources["noise1"].data, 0.5), "noise"
    assert np.allclose(noise_sources["noise2"].data, 1), "noise"


@patch("meegsim.simulate._adjust_snr_local")
def test_simulate_local_snr_adjustment(adjust_snr_mock):
    # return mock PointSource's - 1 noise source, 1 signal source
    simulate_mock = Mock(
        side_effect=[
            [prepare_point_source(name="n1")],
            [prepare_point_source(name="s1")],
        ]
    )

    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    fwd = prepare_forward(5, 4)

    # Define source groups
    source_groups = [
        PointSourceGroup(
            n_sources=1,
            location=[(0, 0)],
            waveform=np.ones((1, 100)),
            snr=np.array([5.0]),
            snr_params=dict(fmin=8, fmax=12),
            std=1,
            names=["s1"],
        ),
    ]
    noise_groups = [
        PointSourceGroup(1, [(1, 1)], np.ones((1, 100)), None, dict(), 1, ["n1"]),
    ]

    with patch.object(
        meegsim.source_groups.PointSourceGroup, "simulate", simulate_mock
    ):
        sfreq = 100
        duration = 5
        times = np.arange(0, sfreq * duration) / sfreq
        _simulate(
            source_groups=source_groups,
            noise_groups=noise_groups,
            coupling_graph=nx.Graph(),
            snr_mode="local",
            snr_global=None,
            snr_params=dict(),
            is_local_snr_adjusted=True,
            src=src,
            times=times,
            fwd=fwd,
            base_std=1e-9,
            random_state=0,
        )

        # Check that the SNR adjustment was performed
        adjust_snr_mock.assert_called()


@patch("meegsim.simulate._adjust_snr_global")
def test_simulate_global_snr_adjustment(adjust_snr_mock):
    # return mock PointSource's - 1 noise source, 1 signal source
    simulate_mock = Mock(
        side_effect=[
            [prepare_point_source(name="n1")],
            [prepare_point_source(name="s1")],
        ]
    )

    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    fwd = prepare_forward(5, 4)

    # Define source groups
    source_groups = [
        PointSourceGroup(
            n_sources=1,
            location=[(0, 0)],
            waveform=np.ones((1, 100)),
            snr=None,
            snr_params=dict(),
            std=1,
            names=["s1"],
        ),
    ]
    noise_groups = [
        PointSourceGroup(1, [(1, 1)], np.ones((1, 100)), None, dict(), 1, ["n1"]),
    ]

    with patch.object(
        meegsim.source_groups.PointSourceGroup, "simulate", simulate_mock
    ):
        sfreq = 100
        duration = 5
        times = np.arange(0, sfreq * duration) / sfreq
        _simulate(
            source_groups=source_groups,
            noise_groups=noise_groups,
            coupling_graph=nx.Graph(),
            snr_mode="global",
            snr_global=5,
            snr_params=dict(fmin=8, fmax=12),
            is_local_snr_adjusted=False,
            src=src,
            times=times,
            fwd=fwd,
            base_std=1e-9,
            random_state=0,
        )

        # Check that the SNR adjustment was performed
        adjust_snr_mock.assert_called()


@patch("meegsim.simulate._set_coupling")
def test_simulate_coupling_setup(set_coupling_mock):
    # return 2 mock PointSource's
    simulate_mock = Mock(
        side_effect=[
            [prepare_point_source(name="s1")],
            [prepare_point_source(name="s2")],
        ]
    )

    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])
    fwd = prepare_forward(5, 4)

    # Define source groups
    source_groups = [
        PointSourceGroup(
            n_sources=2,
            location=[(0, 0), (1, 1)],
            waveform=np.ones((2, 500)),
            snr=None,
            snr_params=dict(),
            std=1,
            names=["s1", "s2"],
        ),
    ]
    noise_groups = []
    coupling = [("s1", "s2", dict(method="ppc_von_mises", kappa=1))]
    coupling_graph = nx.Graph()
    coupling_graph.add_edges_from(coupling)

    with patch.object(
        meegsim.source_groups.PointSourceGroup, "simulate", simulate_mock
    ):
        sfreq = 100
        duration = 5
        times = np.arange(0, sfreq * duration) / sfreq
        _simulate(
            source_groups=source_groups,
            noise_groups=noise_groups,
            coupling_graph=coupling_graph,
            snr_mode="global",
            snr_global=None,
            snr_params=dict(),
            is_local_snr_adjusted=False,
            src=src,
            times=times,
            fwd=fwd,
            base_std=1e-9,
            random_state=0,
        )

        # Check that the coupling setup was performed
        set_coupling_mock.assert_called()


def test_simulate_random_states():
    """
    This test assesses whether the values of random state are passed correctly
    from the most high-level function (_simulate) to the most low-level
    functions (location, waveform, coupling).

    It relies on the mechanism for seed generation used in _simulate:
     - source seeds are generated by SeedSequence from the provided seed
     - coupling seeds are generated by another SeedSequence initiated with the
    last source seed
    """
    # Generate the expected seeds, they will be checked in callables below
    main_seed = 1234
    source_seeds = np.random.SeedSequence(main_seed).generate_state(3)
    coupling_seeds = np.random.SeedSequence(source_seeds[-1]).generate_state(1)

    # first, waveform will be generated for the noise source
    def seed_checking_waveform(n_sources, times, random_state):
        assert random_state == source_seeds[0]
        return np.ones((n_sources, times.size))

    # second, location will be generated for the point source
    def seed_checking_location(src, random_state):
        assert random_state == source_seeds[1]
        return [(0, src[0]["vertno"][0]), (1, src[1]["vertno"][0])]

    # finally, coupling between sources will be generated
    def seed_checking_coupling(waveform, sfreq, random_state):
        assert random_state == coupling_seeds[0]
        return waveform

    # Define parameters and source groups
    sfreq = 100
    duration = 5
    source_groups = [
        PointSourceGroup(
            n_sources=2,
            location=seed_checking_location,
            waveform=np.ones((2, sfreq * duration)),
            snr=None,
            snr_params=dict(),
            std=np.array([1.0, 1.0]),
            names=["s1", "s2"],
        ),
    ]
    noise_groups = [
        PointSourceGroup(
            n_sources=1,
            location=[(0, 0)],
            waveform=seed_checking_waveform,
            snr=None,
            snr_params=dict(),
            std=np.array([1.0]),
            names=["n1"],
        ),
    ]
    coupling = [("s1", "s2", dict(method=seed_checking_coupling))]
    coupling_graph = nx.Graph()
    coupling_graph.add_edges_from(coupling)

    times = np.arange(0, sfreq * duration) / sfreq
    src = prepare_source_space(types=["surf", "surf"], vertices=[[0, 1], [0, 1]])

    _simulate(
        source_groups=source_groups,
        noise_groups=noise_groups,
        coupling_graph=coupling_graph,
        snr_mode="global",
        snr_global=None,
        snr_params=dict(),
        is_local_snr_adjusted=False,
        src=src,
        times=times,
        fwd=None,
        base_std=1e-9,
        random_state=main_seed,
    )

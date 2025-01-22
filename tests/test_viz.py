import mne

from unittest.mock import patch, create_autospec

from meegsim.configuration import SourceConfiguration
from meegsim.viz import plot_source_configuration

from utils.prepare import prepare_source_space


def test_plot_source_configuration_kwargs():
    # Mocking all calls to plotting functions so that a real src is not needed
    brain_mock = create_autospec(mne.viz.get_brain_class())
    stc_mock = create_autospec(mne.SourceEstimate)
    stc_mock.plot.return_value = brain_mock

    src = None
    subject = "meegsim"
    sc = SourceConfiguration(src, sfreq=250, duration=30, random_state=0)

    with patch("meegsim.viz._get_patch_sources_in_hemis", return_value=stc_mock):
        plot_source_configuration(
            sc,
            subject,
            hemi="lh",
            cortex="classic",
            colors=dict(point="red"),
            scale_factors=dict(noise=0.1),
        )

    # Check that cortex="classic" was passed to stc.plot
    stc_plot_call = stc_mock.mock_calls[0]
    assert stc_plot_call.kwargs["cortex"] == "classic", "stc.plot() kwargs"

    # Check that correct values of colors and sizes were used
    # call order: candidate (not here) -> noise -> point
    noise_call, point_call = brain_mock.mock_calls
    assert noise_call.kwargs["scale_factor"] == 0.1, "brain.add_foci() color"
    assert point_call.kwargs["color"] == "red", "brain.add_foci() scale_factor"


def test_plot_source_configuration_kwargs_toggle_source_types():
    # Mocking all calls to plotting functions so that a real src is not needed
    brain_mock = create_autospec(mne.viz.get_brain_class())
    stc_mock = create_autospec(mne.SourceEstimate)
    stc_mock.plot.return_value = brain_mock

    src = prepare_source_space(["surf", "surf"], [[0, 1, 2, 3], [0, 1, 2, 3]])
    subject = "meegsim"
    sc = SourceConfiguration(src, sfreq=250, duration=30, random_state=0)

    with patch("meegsim.viz._get_patch_sources_in_hemis", return_value=stc_mock):
        # Default: point + noise -> 2 calls of brain.add_foci expected
        plot_source_configuration(sc, subject)
        assert len(brain_mock.mock_calls) == 2
        brain_mock.reset_mock()

        # Hide noise sources -> 1
        plot_source_configuration(sc, subject, show_noise_sources=False)
        assert len(brain_mock.mock_calls) == 1
        brain_mock.reset_mock()

        # Show candidate locations -> 3
        plot_source_configuration(sc, subject, show_candidate_locations=True)
        assert len(brain_mock.mock_calls) == 3
        brain_mock.reset_mock()

        # Plot both hemispheres -> 4
        plot_source_configuration(sc, subject)
        assert len(brain_mock.mock_calls) == 2
        brain_mock.reset_mock()

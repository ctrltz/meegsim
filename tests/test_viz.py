from meegsim.configuration import SourceConfiguration
from meegsim.viz import plot_source_configuration


def test_plot_source_configuration():
    # colors/sizes/kwargs can be updated
    # noise/candidate sources are toggled correctly
    # plotting works regardless of whether patch sources are used
    src = None  # most likely, a real src is required
    subject = "fsaverage"
    sc = SourceConfiguration(src, sfreq=250, duration=30, random_state=0)
    plot_source_configuration(sc, subject)

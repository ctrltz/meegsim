import numpy as np
import mne


class MockPointSource:
    """
    Mock PointSource class for testing purposes.
    """
    def __init__(self, name, shape=(1, 100)):
        self.name = name
        self.waveform = np.ones(shape)

    def to_stc(self, *args, **kwargs):
        return mne.SourceEstimate(self.waveform, [[0], []], 0, 0.01)

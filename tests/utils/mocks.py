import numpy as np
import mne


class MockPointSource:
    """
    Mock PointSource class for testing purposes.
    """
    def __init__(self, name, shape=(1, 100)):
        self.name = name
        self.waveform = np.ones(shape)

    @property
    def data(self):
        return np.atleast_2d(self.waveform)
    
    @property
    def vertices(self):
        return np.atleast_2d(np.array([0, 0]))

    def _check_compatibility(self, src):
        # Always compatible
        pass

    def to_stc(self, *args, **kwargs):
        return mne.SourceEstimate(self.waveform, [[0], []], 0, 0.01)

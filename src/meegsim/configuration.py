import numpy as np
import mne

from .sources import _combine_sources_into_stc
from .utils import combine_stcs


class SourceConfiguration:
    """
    Defines a configuration of sources of brain activity and noise.

    Attributes:
    -----------
    src: mne.SourceSpaces
        Source spaces object that stores all candidate source locations.

    sfreq: float
        Sampling frequency of the simulated data, in Hz.

    duration: float
        Length of the simulated data, in seconds.
    """

    # Still need to be added somewhere:
    #  * subject, subjects_dir - might be required by label.center_of_mass
    #    (we could pass as additional arguments to the location function)
    # Q: is `src` in theory subject-specific? need more experience with individual MRI analyses
    #
    # Other ideas:
    #  * saving a configuration
    #  * plotting a configuration with different source groups highlighted in
    #    different colors

    def __init__(self, src, sfreq, duration, random_state=None):
        self.src = src
        
        # Simulation parameters
        self.sfreq = sfreq
        self.duration = duration
        self.n_samples = self.sfreq * self.duration
        self.times = np.arange(self.n_samples) / self.sfreq
        
        # Random state (for reproducibility)
        self.random_state = random_state
        
        # Keep track of all added sources, store 'signal' and 'noise' separately to ease the calculation of SNR
        self._sources = {}
        self._noise_sources = {}

    def check_if_exist(self, names):
        missing = [name for name in names if name not in self._sources]
        if missing:
            raise ValueError(f"The configuration contains no sources with the following names: {', '.join(missing)}")

    def get_waveforms(self, names):
        self.check_if_exist(names)
        waveforms = [self._sources[name].waveform for name in names]
        return np.vstack(waveforms)

    def to_stc(self):
        sources = list(self._sources.values()) 
        noise_sources = list(self._noise_sources.values())
        all_sources = sources + noise_sources

        return _combine_sources_into_stc(all_sources, self.src, self.sfreq)

    def to_raw(self, fwd, info, scaling_factor=1e-6):
        # Parameters:
        # -----------
        #   scaling_factor: float
        #   All source time courses get multiplied by this number before projecting to sensor space.
        #   It looks a bit random in Mina's codes so I think we should attach some physical meaning (e.g., X nA/m) to it.
        
        # Multiply the combined stc by the scaling factor
        stc_combined = self.to_stc() * scaling_factor
    
        # Project to sensor space and return
        raw = mne.apply_forward_raw(fwd, stc_combined, info)
        
        # TODO: add sensor space noise (white noise by default but allow customizing?)
                
        return raw

    def _combine_noise_sources_to_stc(self):
        noise_sources = list(self._noise_sources.values())
        # XXX: might need to provide subject as well
        return _combine_sources_into_stc(noise_sources, self.src, self.sfreq)
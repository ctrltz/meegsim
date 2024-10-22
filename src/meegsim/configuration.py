import numpy as np
import mne

from .sources import _combine_sources_into_stc


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

    random_state: int or None, optional
        Random state that was used to generate the SourceConfiguration.
    """

    def __init__(self, src, sfreq, duration, random_state=None):
        self.src = src
        
        # Simulation parameters
        self.sfreq = sfreq
        self.duration = duration
        self.n_samples = self.sfreq * self.duration
        self.times = np.arange(self.n_samples) / self.sfreq
        self.tstep = self.times[1] - self.times[0]
        
        # Random state (for reproducibility)
        self.random_state = random_state
        
        # Keep track of all added sources, store 'signal' and 'noise' separately to ease the calculation of SNR
        self._sources = {}
        self._noise_sources = {}

    def to_stc(self):
        sources = list(self._sources.values()) 
        noise_sources = list(self._noise_sources.values())
        all_sources = sources + noise_sources

        if not all_sources:
            raise ValueError('No sources were added to the configuration.')

        return _combine_sources_into_stc(all_sources, self.src, self.tstep)

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

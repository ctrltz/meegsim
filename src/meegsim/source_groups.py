"""
This classes store the information about source groups that were
defined by the user until we actually start simulating the data.
"""

from ._check import (
    check_location, check_waveform, 
    check_snr, check_snr_params, check_names
)
from .sources import PointSource


class _BaseSourceGroup:
    def simulate(self):
        raise NotImplementedError(
            'The simulate() method should be implemented in a subclass.'
        )


class PointSourceGroup(_BaseSourceGroup):
    def __init__(
        self, 
        n_sources,
        location, 
        waveform, 
        snr,
        snr_params,
        names
    ):
        super().__init__()

        # Store the defined number of vertices to raise an error 
        # if the output of location function has a different size
        self.n_sources = n_sources

        # Store the provided information
        self.location = location
        self.waveform = waveform
        self.snr = snr
        self.snr_params = snr_params
        self.names = names

    def simulate(self, src, times, random_state=None):
        return PointSource.create(
            src, 
            times,
            self.n_sources,
            self.location,
            self.waveform,
            self.names,
            random_state=random_state
        )
    
    @classmethod
    def create(
        cls,
        src,
        location,
        waveform,
        snr,
        location_params,
        waveform_params,
        snr_params,
        names,
        group,
        existing
    ):
        """
        Check the provided input for all fields and create a source group that
        would store this information.
        """

        location, n_sources = check_location(location, location_params, src)
        waveform = check_waveform(waveform, waveform_params, n_sources)
        snr = check_snr(snr, n_sources)
        snr_params = check_snr_params(snr_params)
        names = check_names(names, group, n_sources, existing)

        return cls(n_sources, location, waveform, snr, snr_params, names)


class PatchSourceGroup(_BaseSourceGroup):
    pass

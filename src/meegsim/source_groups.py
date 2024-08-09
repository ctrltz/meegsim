"""
This classes store the information about source groups that were
defined by the user until we actually start simulating the data.
"""

from ._check import (
    check_location, check_waveform, 
    check_snr, check_snr_params, check_names
)
from .sources import _create_point_sources


class BaseSourceGroup:
    pass


class PointSourceGroup(BaseSourceGroup):
    def __init__(
        self, 
        location, 
        waveform, 
        snr,
        snr_params,
        names
    ):
        super().__init__()

        # Store the provided information
        self.location = location
        self.waveform = waveform
        self.snr = snr
        self.snr_params = snr_params
        self.names = names
        
        # Store the defined number of vertices
        # Raise an error if the output of location function has different size
        self.size = len(names)

    def simulate(self, src, times, fwd=None, random_state=None):
        return _create_point_sources(
            src, 
            times,
            self.location,
            self.waveform,
            self.names,
            random_state=random_state
        )


class PatchSourceGroup(BaseSourceGroup):
    pass


def _create_point_source_group(
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
    location, n_vertices = check_location(location, location_params, src)
    waveform = check_waveform(waveform, waveform_params, n_vertices)
    snr = check_snr(snr, n_vertices)
    snr_params = check_snr_params(snr_params)
    names = check_names(names, group, n_vertices, existing)

    return PointSourceGroup(location, waveform, snr, snr_params, names), names
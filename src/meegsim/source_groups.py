"""
This classes store the information about source groups that were
defined by the user until we actually start simulating the data.
"""

from ._check import (
    check_location, check_waveform, 
    check_snr, check_snr_params, check_names
)
from .sources import PointSource


def generate_names(group, n_sources):
    """
    Automatically generate names for sources belonging to the group.

    Parameters
    ----------
    group: str
        The name of the source group.
    n_sources: int
        The number of sources in the group.

    Returns
    -------
    names: list
        The auto-generated names in the format 'auto-G-sX', where G is the group name,
        and X is the index of the source in the group.
    """
    
    return [f'auto-{group}-s{idx}' for idx in range(n_sources)]


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

    def __repr__(self):
        location_desc = 'list'
        if callable(self.location):
            # extract the name of the function from the partial object if possible
            location_desc = getattr(self.location.func, '__name__', 'callable')
        location_desc = f'location={location_desc}'

        waveform_desc = 'array'
        if callable(self.waveform):
            # extract the name of the function from the partial object if possible
            waveform_desc = getattr(self.waveform.func, '__name__', 'callable')
        waveform_desc = f'waveform={waveform_desc}'

        return f'<PointSourceGroup | {self.n_sources} sources | {location_desc} | {waveform_desc}>'

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

        Parameters
        ----------
        src: mne.SourceSpaces
            The source space that contains all candidate source locations.
        location: list or callable
            The location provided by the user.
        waveform: list of callable
            The waveform provided by the user.
        snr: None, float, or array
            The SNR values provided by the user.
        location_params: dict, optional
            Additional keyword arguments for the location function.
        waveform_params: dict, optional
            Additional keyword arguments for the waveform function.
        snr_params: dict, optional
            Additional parameters for the adjustment of SNR.
        names:
            The names of sources provided by the user.
        group:
            The name of the source group. Only used for generating the names
            of sources automatically.
        existing:
            The names of sources that were already taken.

        Returns
        -------
        group: PointSourceGroup
            A source group object with checked and preprocessed user input.
        """

        # Check the user input
        location, n_sources = check_location(location, location_params, src)
        waveform = check_waveform(waveform, waveform_params, n_sources)
        snr = check_snr(snr, n_sources)
        snr_params = check_snr_params(snr_params, snr)

        # Auto-generate or check the provided source names
        if not names:
            names = generate_names(group, n_sources)
        else:
            check_names(names, n_sources, existing)

        return cls(n_sources, location, waveform, snr, snr_params, names)


class PatchSourceGroup(_BaseSourceGroup):
    pass

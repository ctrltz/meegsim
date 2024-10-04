import networkx as nx

from ._check import check_coupling
from .configuration import SourceConfiguration
from .coupling_graph import _set_coupling
from .source_groups import PointSourceGroup, PatchSourceGroup
from .snr import _adjust_snr
from .waveform import one_over_f_noise


class SourceSimulator:
    """
    This class can be used to create a source configuration by adding 
    point sources (e.g., of narrowband oscillatory activity or 1/f noise.)

    Attributes
    ----------
    src: mne.SourceSpaces
        The source space that contains all candidate source locations.
    """

    def __init__(self, src):
        self.src = src
                       
        # Store groups of sources that were defined with one command
        # Store 'signal' and 'noise' separately to ease the calculation of SNR
        self._source_groups = []
        self._noise_groups = []

        # Keep track of all added sources to check name conflicts
        self._sources = []

        # Store all coupling edges in a graph
        self._coupling_graph = nx.Graph()

        # Keep track whether SNR of any source should be adjusted
        # If yes, then a forward model is required for simulation
        self.is_snr_adjusted = False

    def add_point_sources(
        self, 
        location, 
        waveform, 
        snr=None,
        location_params=dict(), 
        waveform_params=dict(),
        snr_params=dict(),
        names=None
    ):
        """
        Add point sources to the simulation.

        Parameters
        ----------
        location: list or callable
            Locations of sources can be either specified directly as a list of tuples
            (index of the src, vertno) or as a function that returns such a list.
            In the first case, source locations will be the same for every configuration,
            while in the second configurations might differ (e.g., if the function 
            returns a random location).
        waveform: np.array or callable
            Waveforms of source activity provided either directly in an array (fixed
            for every configuration) or as a function that generates the waveforms
            (but differ between configurations if the generation is random). 
        snr: None, float, or array
            SNR values for the defined sources. Can be None (no adjustment of SNR),
            a single value that is used for all sources or an array with one SNR
            value per source.
        location_params: dict, optional
            Keyword arguments that will be passed to the location function.
        waveform_params: dict, optional
            Keyword arguments that will be passed to the waveform function.
        snr_params: dict, optional
            Additional parameters required for the adjustment of SNR.
            Specify fmin and fmax here to define the frequency band which 
            should used for calculating the SNR.
        names: list, optional
            A list of names for each source. If not specified, the names will be
            autogenerated using the format 'sgN-sM', where N is the index of the
            source group, and M is the index of the source in the group.
        
        Returns
        -------
        names: list
            A list of (provided or autogenerated) names for each source
        """
            
        next_group_idx = len(self._source_groups)
        point_sg = PointSourceGroup.create(
            self.src,
            location, 
            waveform, 
            snr=snr,
            location_params=location_params,
            waveform_params=waveform_params,
            snr_params=snr_params,
            names=names,
            group=f'sg{next_group_idx}',
            existing=self._sources
        )
                
        # Store the source group and source names
        self._source_groups.append(point_sg)
        self._sources.extend(point_sg.names)
        
        # Check if SNR should be adjusted
        if point_sg.snr is not None:
            self.is_snr_adjusted = True

        # Return the names of newly added sources
        return point_sg.names
        
    def add_patch_sources(
        self,
        location,
        waveform,
        snr=None,
        location_params=dict(),
        waveform_params=dict(),
        snr_params=dict(),
        extents=None,
        names=None
    ):
        """
        Add point sources to the simulation.

        Parameters
        ----------
        location: list or callable
            Locations of sources can be either specified directly as a list of tuples
            (index of the src, vertno) or as a function that returns such a list.
            In the first case, source locations will be the same for every configuration,
            while in the second configurations might differ (e.g., if the function
            returns a random location).
        waveform: np.array or callable
            Waveforms of source activity provided either directly in an array (fixed
            for every configuration) or as a function that generates the waveforms
            (but differ between configurations if the generation is random). For each vertex in the patch,
            same waveform will be propagated.
        snr: None (do not adjust SNR), float (same SNR for all sources), or array (one value per source)
            TODO: fix when finalizing SNR
            NB: only positive values make sense, raise error if negative ones are provided
        location_params: dict, optional
            Keyword arguments that will be passed to the location function.
        waveform_params: dict, optional
            Keyword arguments that will be passed to the waveform function.
        snr_params: dict, optional
            TODO: fix when finalizing SNR
            fmin and fmax for the frequency band that will be used to adjust SNR.
        extents: list, optional
            Extents (radius in mm) of each patch provided by the user. If None, vertices are selected based on location.
            If a single number, all patch sources have the same extent.
        names: list, optional
            A list of names for each source. If not specified, the names will be
            autogenerated using the format 'sgN-sM', where N is the index of the
            source group, and M is the index of the source in the group.

        Returns
        -------
        names: list
            A list of (provided or autogenerated) names for each source
        """

        next_group_idx = len(self._source_groups)
        patch_sg = PatchSourceGroup.create(
            self.src,
            location,
            waveform,
            snr=snr,
            location_params=location_params,
            waveform_params=waveform_params,
            extents=extents,
            snr_params=snr_params,
            names=names,
            group=f'sg{next_group_idx}',
            existing=self._sources
        )

        # Store the source group and source names
        self._source_groups.append(patch_sg)
        self._sources.extend(patch_sg.names)

        # Return the names of newly added sources
        return patch_sg.names

    def add_noise_sources(
        self, 
        location, 
        waveform=one_over_f_noise, 
        location_params=dict(), 
        waveform_params=dict(),
    ):
        """
        Add noise sources to the simulation. If an adjustment of SNR is needed at
        some point, these sources will be considered as noise.

        Parameters
        ----------
        location: list or callable
            Locations of sources can be either specified directly as a list of tuples
            (index of the src, vertno) or as a function that returns such a list.
            In the first case, source locations will be the same for every configuration,
            while in the second configurations might differ (e.g., if the function 
            returns a random location).
        waveform: np.array or callable
            Waveform provided either directly as an array or as a function.
            By default, 1/f noise is used for all noise sources.
        location_params: dict, optional
            Additional arguments that will be provided to the location function.
            Ignored if location is a list of vertices.
        waveform_params: dict, optional
            Additional arguments that will be provided to the waveform function.
            Ignored if waveform is an array.

        Returns
        -------
        names: list
            Autogenerated names for the noise sources. The format is 'ngN-sM', 
            where N is the index of the noise source group, and M is the index 
            of the source in the group.

        Notes
        -----
        Noise patches are not supported.
        """
        
        next_group_idx = len(self._noise_groups)
        noise_sg = PointSourceGroup.create(
            self.src,
            location,
            waveform, 
            snr=None, 
            location_params=location_params,
            waveform_params=waveform_params,
            snr_params=dict(),
            names=None,
            group=f'ng{next_group_idx}',
            existing=self._sources
        )

        # Store the new source group and source names
        self._noise_groups.append(noise_sg)
        self._sources.extend(noise_sg.names)
        
        # Return the names of newly added sources
        return noise_sg.names
        
    def set_coupling(self, coupling, **common_params):
        """
        Set coupling between sources that were added to the simulator.

        Parameters
        ----------
        coupling: tuple or dict
            Provide a tuple (u, v) to define one pair of coupled sources 
            or a dictionary to define multiple coupling edges at once. u and 
            v are the names of sources that should be coupled. Both 
            sources should be added to the simulation prior to setting the coupling.
        
            If used, the dictionary should contain tuples (u, v) as keys, 
            while the values should be dictionaries with keyword arguments 
            of the coupling method. Use this dictionary to define coupling 
            parameters that are specific for a given edge. Such definitions will 
            also override the common parameters (described below).
        **common_params: dict, optional
            Additional coupling parameters that apply to each edge defined in the 
            coupling dictionary or the single edge if a tuple was provided.

        Notes
        -----
        For the information on required coupling parameters, please refer to the
        :doc:`documentation </api/coupling>` of the corresponding coupling methods.

        Examples
        --------
        Adding a single connectivity edge:

        >>> from meegsim.coupling import ppc_von_mises
        ... 
        ... sim.set_coupling(('s1', 's2'), method=ppc_von_mises, 
        ...                  kappa=1, phase_lag=0, fmin=8, fmax=12)

        Adding multiple connectivity edges at once:
        
        >>> from meegsim.coupling import ppc_von_mises
        ... 
        ... sim.set_coupling(coupling={
        ...     ('s1', 's2'): dict(kappa=1, phase_lag=np.pi/3, fmin=10),
        ...     ('s2', 's3'): dict(kappa=0.5, phase_lag=-np.pi/6)
        ... }, method=ppc_von_mises, fmin=8, fmax=12)

        In the example above, `method` and `fmax` values apply to both 
        coupling edges, while `kappa` and `phase_lag` are edge-specific.
        `fmin` is defined as a common parameter but also has a different
        value for the edge `('s1', 's2')`. Therefore, it will be set to 6 
        for the edge `('s1', 's2')` and to 8 for the edge `('s2', 's3')`.
        """

        # Convert tuple to a dictionary with empty coupling params
        if isinstance(coupling, tuple):
            coupling = {coupling: dict()}

        for coupling_edge, coupling_params in coupling.items():
            params = check_coupling(coupling_edge, coupling_params, common_params, 
                                    self._sources, self._coupling_graph)

            # Add the coupling edge
            source, target = coupling_edge
            self._coupling_graph.add_edge(source, target, **params)
        
    def simulate(
        self,  
        sfreq, 
        duration,
        fwd=None,
        random_state=None
    ):
        """
        Simulate a configuration of defined sources.

        Parameters
        ----------
        sfreq: float
            The sampling frequency of the simulated data, in Hz.
        duration: float
            Duration of the simulated data, in seconds.
        fwd: mne.Forward, optional
            The forward model, only to be used for the adjustment of SNR.
            If no adjustment is performed, the forward model is not required.
        random_state: int or None, optional
            The random state can be provided to obtain reproducible configurations.
            If None (default), the simulated data will differ between function calls.

        Returns
        -------
        sc: SourceConfiguration
            The source configuration, which contains the defined sources and 
            their corresponding waveforms.
        """

        if not (self._source_groups or self._noise_groups):
            raise ValueError('No sources were added to the configuration.')

        if self.is_snr_adjusted and fwd is None:
            raise ValueError('A forward model is required for the adjustment '
                             'of SNR.')

        # Initialize the SourceConfiguration
        sc = SourceConfiguration(self.src, sfreq, duration, random_state=random_state)

        # Simulate signal and noise
        sources, noise_sources = _simulate(
            self._source_groups,
            self._noise_groups,
            self._coupling_graph,
            self.is_snr_adjusted,
            self.src,
            sc.times,
            fwd=fwd,
            random_state=random_state
        )

        # Add the sources to the simulated configuration
        sc._sources = sources
        sc._noise_sources = noise_sources

        return sc


def _simulate(
    source_groups, 
    noise_groups,
    coupling_graph,
    is_snr_adjusted,
    src,
    times,
    fwd,
    random_state=None
):
    """
    This function describes the simulation workflow.
    """
    
    # Simulate all sources independently first (no coupling yet)
    noise_sources = []
    for ng in noise_groups:
        noise_sources.extend(ng.simulate(src, times, random_state=random_state))
    noise_sources = {s.name: s for s in noise_sources}

    sources = []
    for sg in source_groups:
        sources.extend(sg.simulate(src, times, random_state=random_state))
    sources = {s.name: s for s in sources}

    # Setup the desired coupling patterns
    # The time courses are changed for some of the sources in the process
    if coupling_graph.number_of_edges() > 0:
        sources = _set_coupling(sources, coupling_graph, times, random_state=random_state)

    # Adjust the SNR if needed
    if is_snr_adjusted:
        sources = _adjust_snr(src, fwd, sources, source_groups, noise_sources)

    return sources, noise_sources

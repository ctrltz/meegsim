from .configuration import SourceConfiguration
from .source_groups import PointSourceGroup, PatchSourceGroup
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

        # Store all coupling edges
        self._coupling = {}

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
        
    def set_coupling(self, coupling, method):
        raise NotImplementedError('Coupling is not supported yet')
        # coupling = check_coupling(coupling, method)
        # self._coupling.update(coupling)
        
    def simulate(
        self,  
        sfreq, 
        duration,
        random_state=None
    ):
        if not (self._source_groups or self._noise_groups):
            raise ValueError('No sources were added to the configuration.')

        return _simulate(
            self._source_groups,
            self._noise_groups,
            self.src,
            sfreq,
            duration,
            random_state=random_state
        )


def _simulate(
    source_groups, 
    noise_groups,
    src,
    sfreq,
    duration,
    random_state=None
):
    """
    This function describes the simulation workflow.
    """
    # Initialize the SourceConfiguration
    sc = SourceConfiguration(src, sfreq, duration, random_state=random_state)

    # Simulate all sources independently first (no coupling yet)
    noise_sources = []
    for ng in noise_groups:
        noise_sources.extend(ng.simulate(src, sc.times, random_state=random_state))
    noise_sources = {s.name: s for s in noise_sources}

    sources = []
    for sg in source_groups:
        sources.extend(sg.simulate(src, sc.times, random_state=random_state))
    sources = {s.name: s for s in sources}

    # Setup the desired coupling patterns
    # The time courses are changed for some of the sources in the process

        # Here we should also check for possible cycles in the coupling structure
        # If there are cycles, raise an error
        # If there are no cycles, traverse the graph and set coupling according to the selected method
        # Try calling the coupling with the provided parameters but be prepared for mismatches

    # Adjust the SNR of sources in each source group
    # 1. Estimate the noise variance in the specified band
    # fwd_noise = mne.forward.restrict_forward_to_stc(fwd, stc_noise, on_missing='raise')
    # noise_var = get_sensor_space_variance()
    # 2. Adjust the amplitude of each signal source according to the desired SNR (if not None)

    # Add the sources to the simulated configuration
    sc._sources = sources
    sc._noise_sources = noise_sources

    return sc

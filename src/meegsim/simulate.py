import networkx as nx

from ._check import check_coupling, check_option, check_numeric_array, check_snr_params
from .configuration import SourceConfiguration
from .coupling_graph import _set_coupling
from .source_groups import PointSourceGroup, PatchSourceGroup
from .snr import _adjust_snr_local, _adjust_snr_global
from .waveform import one_over_f_noise


class SourceSimulator:
    """
    This class can be used to create configurations of sources
    of activity (e.g., narrowband oscillation or 1/f noise).

    Attributes
    ----------
    src : mne.SourceSpaces
        The source space that contains all candidate source locations.
    snr_mode : {'global', 'local'}
        The desired mode for the adjustment of the signal-to-noise ratio (SNR).

        .. note::

          * If ``'global'`` (default), the total power of **all** point/patch sources is
            adjusted relative to the total power of **all** noise sources. The target
            value of SNR should be provided in the ``snr_global`` argument of
            :meth:`~meegsim.simulate.SourceSimulator.simulate`. The spatial distribution
            of source activity can be controlled using the ``std`` argument when adding
            sources.

          * If ``'local'``, the power of **each** point/patch source is adjusted relative
            to the total power of **all** noise sources. The target value(s) of SNR
            should be provided in the ``snr`` argument when adding sources to the
            simulation via either
            :meth:`~meegsim.simulate.SourceSimulator.add_point_sources` or
            :meth:`~meegsim.simulate.SourceSimulator.add_patch_sources`
    base_std : float, optional
        The source activity is scaled by the base standard deviation before
        projecting to sensor space. By default, it is equal to :math:`10^{-9}`,
        corresponding to the dipolar moment of 1 nAm.
    """

    def __init__(self, src, snr_mode="global", base_std=1e-9):
        self.src = src
        self.base_std = base_std

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
        self.is_local_snr_adjusted = False

        # Global adjustment of SNR: combined signal vs. combined noise
        # Local adjustment of SNR: each signal separately vs. combined noise
        self.snr_mode = check_option("snr_mode", snr_mode, ["global", "local"])

    def add_point_sources(
        self,
        location,
        waveform,
        snr=None,
        std=1.0,
        location_params=dict(),
        waveform_params=dict(),
        snr_params=dict(),
        names=None,
    ):
        """
        Add point sources to the simulation.

        Parameters
        ----------
        location : list or callable
            Locations of sources can be either specified directly as a list of tuples
            (index of the src, vertno) or as a function that returns such a list.
            In the first case, source locations will be the same for every configuration.
            In the second case, configurations might differ (e.g., if the function
            returns a random location).
        waveform : array or callable
            Waveforms of source activity provided either directly in an array (fixed
            for every configuration) or as a function that generates the waveforms
            (but differ between configurations if the generation is random).
        snr : None, float, or array, optional
            SNR values for the defined sources, only used if ``snr_mode`` is set to
            ``'local'``. Can be None (no adjustment of SNR), a single value
            that is used for all sources or an array with one SNR
            value per source.
        std : float or array, optional
            Desired standard deviation of the source activity, provided either as a
            single value that applies to all sources or as an array with one value per
            source. This parameter can be used in combination with the global SNR
            mode to set an arbitrary spatial distribution of source activity.
            By default, 1 is used so the variance of all sources is the same.
            If the value of local SNR is specified, this parameter will effectively
            be ignored.
        location_params : dict, optional
            Keyword arguments that will be passed to ``location``
            if a function is provided.
        waveform_params : dict, optional
            Keyword arguments that will be passed to ``waveform``
            if a function is provided.
        snr_params : dict, optional
            Additional parameters required for the adjustment of SNR.
            Specify ``fmin`` and ``fmax`` here to define the frequency band which
            should used for calculating the SNR.
        names : list, optional
            A list of names for each source. If not specified, the names will be
            autogenerated using the format 'auto-sgN-sM', where N is the index
            of the source group, and M is the index of the source in the group.

        Returns
        -------
        names : list
            A list of (provided or autogenerated) names for each source.
        """

        next_group_idx = len(self._source_groups)
        point_sg = PointSourceGroup.create(
            self.src,
            location,
            waveform,
            snr=snr,
            std=std,
            location_params=location_params,
            waveform_params=waveform_params,
            snr_params=snr_params,
            names=names,
            group=f"sg{next_group_idx}",
            existing=self._sources,
        )

        # Store the source group and source names
        self._source_groups.append(point_sg)
        self._sources.extend(point_sg.names)

        # Check if SNR should be adjusted
        if point_sg.snr is not None and self.snr_mode == "local":
            self.is_local_snr_adjusted = True

        # Return the names of newly added sources
        return point_sg.names

    def add_patch_sources(
        self,
        location,
        waveform,
        snr=None,
        std=1.0,
        location_params=dict(),
        waveform_params=dict(),
        snr_params=dict(),
        extents=None,
        names=None,
    ):
        """
        Add patch sources to the simulation.

        Parameters
        ----------
        location : list or callable
            Locations of sources can be either specified directly as a list of tuples
            (index of the src, vertno) or as a function that returns such a list.
            In the first case, source locations will be the same for every configuration.
            In the second case, configurations might differ (e.g., if the function
            returns a random location).

            Depending on the value of the ``extents`` argument, locations are
            interpreted either as set of vertices belonging to the patch (default)
            or centers of patches.
        waveform : array or callable
            Waveforms of source activity provided either directly in an array (fixed
            for every configuration) or as a function that generates the waveforms
            (might differ between configurations if the generation is random).
            For each vertex in the patch, the same waveform is currently used.
        snr : None (default), float, or array
            SNR values for the defined sources, only used if ``snr_mode`` is set to
            ``'local'``. Can be None (no adjustment of SNR, default),
            a single value that is used for all sources or an array
            with one SNR value per source.
        std : float or array, optional
            Desired standard deviation of the source activity, provided either as a
            single value that applies to all sources or as an array with one value per
            source. This parameter can be used in combination with the global SNR
            mode to set an arbitrary spatial distribution of source activity.
            By default, 1 is used so the variance of all sources is the same.
            If the value of local SNR is specified, this parameter will effectively
            be ignored.
        location_params : dict, optional
            Keyword arguments that will be passed to ``location`` if a
            function is provided.
        waveform_params : dict, optional
            Keyword arguments that will be passed to ``waveform`` if a
            function is provided.
        snr_params : dict, optional
            Additional parameters required for the adjustment of SNR.
            Specify ``fmin`` and ``fmax`` here to define the frequency band which
            should used for calculating the SNR.
        extents : None (default), float, or list
            Extents (radius, in mm) of each patch. If None (default), location must
            contain all vertices belonging to the patch(es). If specified, patch are
            grown (using :func:`mne.grow_labels`) from vertices specified in
            location according to the provided values of extent. If a single number
            is provided, all patch sources have the same extent.
        names : list, optional
            A list of names for each source. If not specified, the names will be
            autogenerated using the format 'auto-sgN-sM', where N is the index
            of the source group, and M is the index of the source in the group.

        Returns
        -------
        names : list
            A list of (provided or autogenerated) names for each source
        """

        next_group_idx = len(self._source_groups)
        patch_sg = PatchSourceGroup.create(
            self.src,
            location,
            waveform,
            snr=snr,
            std=std,
            location_params=location_params,
            waveform_params=waveform_params,
            extents=extents,
            snr_params=snr_params,
            names=names,
            group=f"sg{next_group_idx}",
            existing=self._sources,
        )

        # Store the source group and source names
        self._source_groups.append(patch_sg)
        self._sources.extend(patch_sg.names)

        # Check if SNR should be adjusted
        if patch_sg.snr is not None and self.snr_mode == "local":
            self.is_local_snr_adjusted = True

        # Return the names of newly added sources
        return patch_sg.names

    def add_noise_sources(
        self,
        location,
        waveform=one_over_f_noise,
        std=1.0,
        location_params=dict(),
        waveform_params=dict(),
    ):
        """
        Add noise sources to the simulation. If an adjustment of SNR is needed at
        some point, these sources will be considered as noise.

        Parameters
        ----------
        location : list or callable
            Locations of sources can be either specified directly as a list of tuples
            (index of the src, vertno) or as a function that returns such a list.
            In the first case, source locations will be the same for every configuration.
            In the second case, configurations might differ (e.g., if the function
            returns a random location).
        waveform : array or callable
            Waveform provided either directly as an array or as a function.
            By default, 1/f noise with the slope of 1 is used for all noise sources.
        std : float or array, optional
            Desired standard deviation of the source activity, provided either as a
            single value that applies to all sources or as an array with one value per
            source. By default, 1 is used so the variance of all noise sources is
            the same.
        location_params : dict, optional
            Keyword arguments that will be passed to ``location`` if a
            function is provided.
        waveform_params : dict, optional
            Keyword arguments that will be passed to ``waveform`` if a
            function is provided.

        Returns
        -------
        names : list
            Autogenerated names for the noise sources. The format is 'auto-ngN-sM',
            where N is the index of the noise source group, and M is the index
            of the source in the group.

        Notes
        -----
        Noise patches are currently not supported.
        """

        next_group_idx = len(self._noise_groups)
        noise_sg = PointSourceGroup.create(
            self.src,
            location,
            waveform,
            snr=None,
            std=std,
            location_params=location_params,
            waveform_params=waveform_params,
            snr_params=dict(),
            names=None,
            group=f"ng{next_group_idx}",
            existing=self._sources,
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
        coupling : tuple or dict
            Provide a tuple ``(u, v)`` to define one pair of coupled sources
            or a dictionary to define multiple coupling edges at once. ``u`` and
            ``v`` are the names of sources that should be coupled. Both
            sources should be added to the simulation prior to setting the coupling.

            If used, the dictionary should contain tuples ``(u, v)`` as keys,
            while the values should be dictionaries with keyword arguments
            of the coupling method. Use this dictionary to define coupling
            parameters that are specific for a given edge. Such definitions will
            also override the common parameters (described below).
        **common_params : dict, optional
            Additional coupling parameters that apply to each edge defined in the
            coupling dictionary or the single edge if a tuple was provided.

        Notes
        -----
        For the information on required coupling parameters, please refer to the
        :doc:`documentation </api/coupling>` of the corresponding coupling method(s).

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

        In the example above, ``method`` and ``fmax`` values apply to both
        coupling edges, while ``kappa`` and ``phase_lag`` are edge-specific.
        ``fmin`` is defined as a common parameter but also has a different
        value for the edge ``('s1', 's2')``. Therefore, it will be set to `6`
        for the edge ``('s1', 's2')`` and to `8` for the edge ``('s2', 's3')``.
        """

        # Convert tuple to a dictionary with empty coupling params
        if isinstance(coupling, tuple):
            coupling = {coupling: dict()}

        for coupling_edge, coupling_params in coupling.items():
            params = check_coupling(
                coupling_edge,
                coupling_params,
                common_params,
                self._sources,
                self._coupling_graph,
            )

            # Add the coupling edge
            source, target = coupling_edge
            self._coupling_graph.add_edge(source, target, **params)

    def simulate(
        self,
        sfreq,
        duration,
        fwd=None,
        snr_global=None,
        snr_params=dict(),
        random_state=None,
    ):
        """
        Simulate a configuration of defined sources.

        Parameters
        ----------
        sfreq : float
            The sampling frequency of the simulated data, in Hz.
        duration : float
            Duration of the simulated data, in seconds.
        fwd : None or Forward, optional
            The forward model, only to be used for the adjustment of SNR.
            If no adjustment is performed, the forward model is not required.
        snr_global : float or None, optional
            The value of global SNR, only used if the ``snr_mode`` is set to
            ``'global'``. If None (default), no adjustment of global SNR is performed.
        snr_params : dict, optional
            Additional parameters required for the adjustment of global SNR.
            Specify ``fmin`` and ``fmax`` here to define the frequency band which
            should used for calculating the SNR.
        random_state : int or None, optional
            The random state can be provided to obtain reproducible configurations.
            If None (default), the simulated data will differ between function calls.

        Returns
        -------
        sc : SourceConfiguration
            The source configuration, which contains the defined sources and
            their corresponding waveforms.
        """

        if not (self._source_groups or self._noise_groups):
            raise ValueError("No sources were added to the configuration.")

        # We expect None or one value that applies to all sources
        snr_global = check_numeric_array(
            "global SNR", snr_global, n_sources=1, bounds=(0, None), allow_none=True
        )
        snr_params = check_snr_params(snr_params, snr_global)

        # Check the forward model and auto-fill info if needed
        is_global_snr_adjusted = self.snr_mode == "global" and snr_global is not None
        is_local_snr_adjusted = self.snr_mode == "local" and self.is_local_snr_adjusted
        if (is_global_snr_adjusted or is_local_snr_adjusted) and fwd is None:
            raise ValueError("A forward model is required for the adjustment of SNR.")

        # Initialize the SourceConfiguration
        sc = SourceConfiguration(self.src, sfreq, duration, random_state=random_state)

        # Simulate signal and noise
        sources, noise_sources = _simulate(
            source_groups=self._source_groups,
            noise_groups=self._noise_groups,
            coupling_graph=self._coupling_graph,
            snr_mode=self.snr_mode,
            snr_global=snr_global,
            snr_params=snr_params,
            is_local_snr_adjusted=self.is_local_snr_adjusted,
            src=self.src,
            times=sc.times,
            fwd=fwd,
            base_std=self.base_std,
            random_state=random_state,
        )

        # Add the sources to the simulated configuration
        sc._sources = sources
        sc._noise_sources = noise_sources

        return sc


def _simulate(
    source_groups,
    noise_groups,
    coupling_graph,
    snr_mode,
    snr_global,
    snr_params,
    is_local_snr_adjusted,
    src,
    times,
    fwd,
    base_std,
    random_state=None,
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
        _set_coupling(sources, coupling_graph, times, random_state=random_state)

    # Set the standard deviation of all sources w.r.t. base std
    # NOTE: this should also be helpful to get less warnings about unreasonably
    # high values of source activity from apply_forward_raw
    for s in sources.values():
        s.waveform *= base_std * s.std
    for s in noise_sources.values():
        s.waveform *= base_std * s.std

    # Adjust the SNR if needed
    if snr_mode == "global" and snr_global is not None:
        tstep = times[1] - times[0]
        _adjust_snr_global(
            src, fwd, snr_global, snr_params, tstep, sources, noise_sources
        )
    elif is_local_snr_adjusted:
        tstep = times[1] - times[0]
        _adjust_snr_local(src, fwd, tstep, sources, source_groups, noise_sources)

    return sources, noise_sources

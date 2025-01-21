import numpy as np
import mne

from ._check import check_numeric
from .sensor_noise import _adjust_sensor_noise, _prepare_sensor_noise
from .sources import _combine_sources_into_stc
from .viz import plot_source_configuration


class SourceConfiguration:
    """
    This class describes a simulated configuration of sources
    of brain activity and noise.

    Attributes
    ----------
    src : SourceSpaces
        Source spaces object that stores all candidate source locations.

    sfreq : float
        Sampling frequency of the simulated data, in Hz.

    duration : float
        Length of the simulated data, in seconds.

    random_state : int or None, optional
        Random state that was used to generate the configuration.
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

    def plot(
        self,
        subject,
        hemi,
        colors=None,
        sizes=None,
        show_noise_sources=True,
        show_candidate_locations=False,
        **brain_kwargs,
    ):
        return plot_source_configuration(
            self,
            subject,
            hemi,
            colors,
            sizes,
            show_noise_sources,
            show_candidate_locations,
            **brain_kwargs,
        )

    def to_stc(self):
        """
        Obtain an ``stc`` object that contains data from all sources
        in the configuration.

        Returns
        -------
        stc : SourceEstimate
            The resulting stc object that contains data from all sources.
        """
        sources = list(self._sources.values())
        noise_sources = list(self._noise_sources.values())
        all_sources = sources + noise_sources

        if not all_sources:
            raise ValueError("No sources were added to the configuration.")

        return _combine_sources_into_stc(all_sources, self.src, self.tstep)

    def to_raw(self, fwd, info, scaling_factor=1e-6, sensor_noise_level=None):
        """
        Project the activity of all simulated sources to sensor space.

        Parameters
        ----------
        fwd : Forward
            The forward model.
        info : Info
            The info structure that describes the channel layout.
        scaling_factor : float, optional
            The source activity is scaled by this factor before projecting to
            sensor space. By default, the scaling factor is equal to :math:`10^{-6}`.
        sensor_noise_level : float, optional
            The desired level of sensor-space noise between 0 and 1. For example,
            if 0.1 is specified, 10% of total sensor-space power will stem from
            white noise with an identity covariance matrix, while the remaining 90%
            of power will be explained by source activity projected to sensor space.
            By default, no sensor space noise is added. See Notes for more details.

        Returns
        -------
        raw : Raw
            The simulated sensor space data.

        Notes
        -----
        The adjustment of sensor space noise is performed as follows:

        1. The sensor space noise is scaled to equalize the mean sensor-space variance
        of broadband noise and brain activity.

        2. The brain activity and noise are mixed to achieve the desired level of
        sensor space noise (denoted by :math:`\\gamma` below):

        .. math::

            \\begin{eqnarray}
                y & = \\sqrt{1 - \\gamma} \\cdot y_{brain} + \\sqrt{\\gamma} \\cdot y_{noise} \\\\
                \\\\
                P_{total} & = (1 - \\gamma) \\cdot P_{brain} + \\gamma \\cdot P_{noise}
            \\end{eqnarray}
        """
        check_numeric("sensor_noise_level", sensor_noise_level, [0.0, 1.0])

        # Multiply the combined stc by the scaling factor
        stc_combined = self.to_stc() * scaling_factor

        # Project to sensor space and return
        raw = mne.apply_forward_raw(fwd, stc_combined, info)

        # Add sensor space noise if needed
        if sensor_noise_level:
            noise = _prepare_sensor_noise(raw, self.times, self.random_state)
            raw = _adjust_sensor_noise(raw, noise, sensor_noise_level)

        return raw

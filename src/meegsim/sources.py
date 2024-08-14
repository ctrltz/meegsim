"""
Classes that store all information about simulated sources and their groups.
Advantage of this approach over stc: in stc, we can only have one time series per vertex, 
so if point sources coincide or patches overlap, we lose access to the original time series.

Information to store:
    * src_idx - which hemisphere or source space the source belongs to
    * location, location_params - as provided by the user
    * waveform, waveform_params - as provided by the user
    * snr, snr_params - as provided by the user
    * _data - actual generated time series
    * _vertices - actual generated positions (vertno) of sources

    * patch_corr - correlation between vertices of the patch
	
Methods:
	* generate - executes location and waveform functions if needed
	* data (property) - return self._data (if needed, calls generate before that)
	* vertices (property) - returns self._vertices (if needed, calls generate before that)
	* to_stc (property) - ready-to-use mne.SourceEstimate object with data and vertices properly filled
"""

import numpy as np
import mne

from .utils import combine_stcs


class BaseSource:
    def __init__(self, waveform):        
        # Current constraint: one source corresponds to one waveform
        # Point source: the waveform is present in one vertex
        # Patch source: the waveform is mixed with noise in several vertices
        # TODO: how to proceed free dipole orientations?
        self.waveform = waveform


class PointSource(BaseSource):
    def __init__(self, src_idx, vertno, waveform):
        super().__init__(waveform)

        self.src_idx = src_idx
        self.vertno = vertno

    def __repr__(self):
        return f'<PointSource | src={self.src_idx} | vertno={self.vertno}>'

    def to_stc(self, src, sfreq, subject=None):
        data = self.waveform[np.newaxis, :]
        # TODO: won't work with fancy srcs
        vertices = [[], []]
        vertices[self.src_idx].append(self.vertno)

        return mne.SourceEstimate(
            data=data,
            vertices=vertices,
            tmin=0,
            tstep=1.0 / sfreq,
            subject=subject
        )


class PatchSource(BaseSource):
    def __init__(self, patch_corr):
        self._label = None
        self.patch_corr = patch_corr


def _create_point_sources(
    src,
    times,
    location,
    waveform,
    names,
    random_state=None,
):
    """
    This function creates point sources according to the provided input.
    """

    # Get the list of vertices (directly from the provided input or through the function)
    vertices = location(src, random_state=random_state) if callable(location) else location
    n_vertices = sum([len(vs) for vs in vertices])

    # what if custom location functions produce non-unique values? we might need to warn the user or forbid it

    # Get the corresponding number of time series
    data = waveform(n_vertices, times, random_state=random_state) if callable(waveform) else waveform
        
    # Create point sources and save them as a group
    sources = []
    for (src_idx, vertno), waveform, name in zip(vertices, data, names):
        sources.append(PointSource(name, src_idx, vertno, waveform))
        
    return sources


def _combine_sources_into_stc(sources, src, sfreq):
    stc_combined = None
    
    for s in sources:
        stc_source = s.to_stc(src, sfreq)
        if stc_combined is None:
            stc_combined = stc_source
            continue

        stc_combined = combine_stcs(stc_combined, stc_source)

    return stc_combined
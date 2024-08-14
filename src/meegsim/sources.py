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

from .utils import combine_stcs, _extract_hemi


class BaseSource:
    """
    An abstract class representing a source of activity.
    """

    def __init__(self, waveform):        
        # Current constraint: one source corresponds to one waveform
        # Point source: the waveform is present in one vertex
        # Patch source: the waveform is mixed with noise in several vertices
        self.waveform = waveform

    def to_stc(self):
        raise NotImplementedError(
            'The to_stc() method should be implemented in the subclass.'
        )


class PointSource(BaseSource):
    def __init__(self, src_idx, vertno, waveform, hemi=None):
        super().__init__(waveform)

        self.src_idx = src_idx
        self.vertno = vertno
        self.hemi = hemi

    def __repr__(self):
        # Use human readable names of hemispheres if possible
        src_desc = self.hemi if self.hemi else f'src[{self.src_idx}]'
        return f'<PointSource | {src_desc} | {self.vertno}>'

    def to_stc(self, src, sfreq, subject=None):
        if self.src_idx >= len(src):
            raise ValueError(
                f"The point source cannot be added to the provided src. "
                f"The point source was assigned to source space {self.src_idx}, "
                f"which is not present in the provided src object."
            )
        
        if self.vertno not in src[self.src_idx]['vertno']:
            raise ValueError(
                f"The point source cannot be added to the provided src. "
                f"The source space with index {self.src_idx} does not "
                f"contain the vertex {self.vertno}"
            )

        data = self.waveform[np.newaxis, :]
        
        # Create a list of vertices for each src
        vertices = [[] for _ in src]
        vertices[self.src_idx].append(self.vertno)

        return mne.SourceEstimate(
            data=data,
            vertices=vertices,
            tmin=0,
            tstep=1.0 / sfreq,
            subject=subject
        )


class PatchSource(BaseSource):
    def __init__(self):
        pass


def _create_point_sources(
    src, 
    times, 
    location, 
    waveform, 
    snr=None, 
    location_params=None, 
    waveform_params=None, 
    random_state=None, 
    names=None,
    name_prefix=''
):
    """
    This function creates point sources according to the provided input.
    """

    # Get the list of vertices (directly from the provided input or through the function)
    vertices = location(src, random_state=random_state, **location_params) if callable(location) else location
    # TODO: check that the format of vertices matches src	  
    n_vertices = sum([len(vs) for vs in vertices])

    # Check the provided names, broadcast to match the number of vertices
    if names is None:
        names = [None] * n_vertices
    if len(names) != n_vertices:
        raise ValueError('The number of provided source names does not match the number of generated source locations')

    # Get the corresponding number of time series
    data = waveform(n_vertices, times, random_state=random_state, **waveform_params) if callable(waveform) else waveform
    if data.shape != (n_vertices, len(times)):
        # TODO: split into two errors with more informative messages?
        raise ValueError('The provided array/function for source waveform does not match other simulation parameters')

    # Here we should also adjust the SNR
    # 2. Adjust the amplitude of each signal source (self._sources) according to the desired SNR (if not None)
        
    # Create point sources and save them as a group
    sources = {}
    for (src_idx, vertno), waveform, name in zip(vertices, data, names):
        hemi = _extract_hemi(src[src_idx])
        new_source = PointSource(src_idx, vertno, waveform, hemi=hemi)

        if name is None:
            src_desc = hemi if hemi else f'src{src_idx}'
            name = f'{name_prefix}{src_desc}-{vertno}'
        sources[name] = new_source
        
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
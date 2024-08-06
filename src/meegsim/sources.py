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

from .utils import _extract_hemi
# from .utils import src_vertno_to_vertices


class BaseSource:
    def __init__(self, waveform):        
        # Current constraint: one source corresponds to one waveform
        # Point source: the waveform is present in one vertex
        # Patch source: the waveform is mixed with noise in several vertices
        self.waveform = waveform


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

    # Flatten the vertices list, keep the source space indices in a separate list
    src_indices = []
    hemis = []
    vertices_flat = []
    for src_idx, src_vertno in enumerate(vertices):
        n_vertno = len(src_vertno)
        src_indices.extend([src_idx] * n_vertno)
        hemi = _extract_hemi(src[src_idx])
        hemis.extend([hemi] * n_vertno)
        vertices_flat.extend(src_vertno)
        
    # Create point sources and save them as a group
    sources = {}
    for src_idx, hemi, vertno, waveform, name in zip(src_indices, hemis, vertices_flat, data, names):
        new_source = PointSource(src_idx, vertno, waveform, hemi=hemi)

        if name is None:
            src_desc = hemi if hemi else f'src{src_idx}'
            name = f'{name_prefix}{src_desc}-{vertno}'
        sources[name] = new_source
        
    return sources
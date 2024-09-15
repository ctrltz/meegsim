"""
Classes that store all information about simulated sources.
Advantage of this approach over stc: in stc, we can only have one time series 
per vertex, so if point sources coincide or patches overlap, we lose access 
to the original time series.
"""

import numpy as np
import mne

from .utils import combine_stcs, get_sfreq, _extract_hemi


class _BaseSource:
    """
    An abstract class representing a source of activity.
    """

    def __init__(self, waveform, sfreq):        
        # Current constraint: one source corresponds to one waveform
        # Point source: the waveform is present in one vertex
        # Patch source: the waveform is mixed with noise in several vertices
        self.waveform = waveform
        self.sfreq = sfreq

    def to_stc(self):
        raise NotImplementedError(
            'The to_stc() method should be implemented in the subclass.'
        )


class PointSource(_BaseSource):
    """
    Point source of activity that is located in one of the vertices in
    the source space.

    Attributes
    ----------
    src_idx: int
        The index of source space that the point source belong to.
    vertno: int
        The vertex that the point source correspond to
    waveform: np.array
        The waveform of source activity.
    sfreq: float
        The sampling frequency of the activity time course.
    hemi: str or None, optional
        Human-readable name of the hemisphere (e.g, lh or rh).
    """

    def __init__(self, name, src_idx, vertno, waveform, sfreq, hemi=None):
        super().__init__(waveform, sfreq)

        self.name = name
        self.src_idx = src_idx
        self.vertno = vertno
        self.sfreq = sfreq
        self.hemi = hemi

    def __repr__(self):
        # Use human readable names of hemispheres if possible
        src_desc = self.hemi if self.hemi else f'src[{self.src_idx}]'
        return f'<PointSource | {self.name} | {src_desc} | {self.vertno}>'

    def to_stc(self, src, subject=None):
        """
        Convert the point source into a SourceEstimate object in the context
        of the provided SourceSpaces.

        Parameters
        ----------
        src: mne.SourceSpaces
            The source space where the point source should be considered.
        subject: str or None, optional
            Name of the subject that the stc corresponds to.
            If None, the subject name from the provided src is used if present.
        
        Returns
        -------
        stc: mne.SourceEstimate
            SourceEstimate that corresponds to the provided src and contains 
            one active vertex.

        Raises
        ------
        ValueError
            If the point source does not exist in the provided src.
        """

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

        # Resolve the subject name as done in MNE
        if subject is None:
            subject = src[0].get("subject_his_id", None)

        data = self.waveform[np.newaxis, :]
        
        # Create a list of vertices for each src
        vertices = [[] for _ in src]
        vertices[self.src_idx].append(self.vertno)

        return mne.SourceEstimate(
            data=data,
            vertices=vertices,
            tmin=0,
            tstep=1.0 / self.sfreq,
            subject=subject
        )

    @classmethod
    def create(
        cls,
        src,
        times,
        n_sources,
        location,
        waveform,
        names,
        random_state=None
    ):
        """
        This function creates point sources according to the provided input.
        """

        # Get the sampling frequency
        sfreq = get_sfreq(times)

        # Get the list of vertices (directly from the provided input or through the function)
        vertices = location(src, random_state=random_state) if callable(location) else location
        if len(vertices) != n_sources:
            raise ValueError('The number of sources in location does not match')

        # Get the corresponding number of time series
        data = waveform(n_sources, times, random_state=random_state) if callable(waveform) else waveform
        if data.shape[0] != n_sources:
            raise ValueError('The number of sources in waveform does not match')
        if data.shape[1] != len(times):
            raise ValueError('The number of samples in waveform does not match')

        # Create point sources and save them as a group
        sources = []
        for (src_idx, vertno), waveform, name in zip(vertices, data, names):
            hemi = _extract_hemi(src[src_idx])
            sources.append(cls(
                name=name, 
                src_idx=src_idx, 
                vertno=vertno, 
                waveform=waveform, 
                sfreq=sfreq, 
                hemi=hemi
            ))
            
        return sources        


class PatchSource(_BaseSource):
    """
    Patch source of activity that is located in one of the vertices in
    the source space.

    Attributes
    ----------
    src_idx: int
        The index of source space that the patch source belong to.
    vertno: int
        The central vertex of  the patch.
    patch_vertno: list
        The vertices that the patch sources correspond to including the central vertex.
    waveform: np.array
        The waveform of source activity.
    sfreq: float
        The sampling frequency of the activity time course.
    extent: float
        Extents (radius in mm) of the patch, defines the number of vertices in patch_vertno.
    hemi: str or None, optional
        Human-readable name of the hemisphere (e.g, lh or rh).
    """

    def __init__(self, name, src_idx, vertno, patch_vertno, waveform, sfreq, extent=None, hemi=None):
        super().__init__(waveform, sfreq)

        self.name = name
        self.src_idx = src_idx
        self.vertno = vertno
        self.patch_vertno = patch_vertno
        self.sfreq = sfreq
        self.extent = extent
        self.hemi = hemi

    def __repr__(self):
        # Use human readable names of hemispheres if possible
        src_desc = self.hemi if self.hemi else f'src[{self.src_idx}]'
        return f'<PatchSource | {self.name} | {src_desc} | {self.vertno} | {self.patch_vertno}>'

    def to_stc(self, src, subject=None):
        """
        Convert the patch source into a SourceEstimate object in the context
        of the provided SourceSpaces.

        Parameters
        ----------
        src: mne.SourceSpaces
            The source space where the patch source should be considered.
        subject: str or None, optional
            Name of the subject that the stc corresponds to.
            If None, the subject name from the provided src is used if present.

        Returns
        -------
        stc: mne.SourceEstimate
            SourceEstimate that corresponds to the provided src and contains
            one active vertex.

        Raises
        ------
        ValueError
            If the patch source does not exist in the provided src.
        """

        if self.src_idx >= len(src):
            raise ValueError(
                f"The patch source cannot be added to the provided src. "
                f"The patch source was assigned to source space {self.src_idx}, "
                f"which is not present in the provided src object."
            )

        if self.vertno not in src[self.src_idx]['vertno']:
            raise ValueError(
                f"The patch source cannot be added to the provided src. "
                f"The source space with index {self.src_idx} does not "
                f"contain the vertex {self.vertno}"
            )

        # Resolve the subject name as done in MNE
        if subject is None:
            subject = src[0].get("subject_his_id", None)

        # Create a list of vertices for each src
        vertices = [[] for _ in src]

        if self.extent is not None:
            vertices[self.src_idx].extend(self.patch_vertno)
            data = np.tile(self.waveform[np.newaxis, :], (len(self.patch_vertno), 1))
        else:  # if locations ia a label
            vertices[self.src_idx].append(self.vertno)
            data = self.waveform

        return mne.SourceEstimate(
            data=data,
            vertices=vertices,
            tmin=0,
            tstep=1.0 / self.sfreq,
            subject=subject
        )

    @classmethod
    def create(
            cls,
            src,
            times,
            n_sources,
            location,
            waveform,
            names,
            extents,
            random_state=None
    ):
        """
        This function creates patch sources according to the provided input.
        """

        # Get the sampling frequency
        sfreq = get_sfreq(times)

        # Get the list of vertices (directly from the provided input or through the function)
        vertices = location(src, random_state=random_state) if callable(location) else location
        if len(vertices) != n_sources:
            raise ValueError('The number of sources in location does not match')

        # Get the corresponding number of time series
        data = waveform(n_sources, times, random_state=random_state) if callable(waveform) else waveform
        if data.shape[0] != n_sources:
            raise ValueError('The number of sources in waveform does not match')
        if data.shape[1] != len(times):
            raise ValueError('The number of samples in waveform does not match')

        # check if extents is a list, otherwise make it a list
        if not isinstance(extents, list):
            extents = [extents]

        # find patch vertices
        subject = src[0].get("subject_his_id", None)
        patch_vertices = []
        for isource, extent in enumerate(extents):
            if extent is not None:
                src_idx = vertices[isource][0]
                vertno = vertices[isource][1]
                patch = mne.grow_labels(subject, vertno, extent, src_idx, subjects_dir=None)[0]
                # prune vertices
                patch_vertno = [vert for vert in patch.vertices if vert in src[src_idx]['vertno']]
                patch_vertices.append(patch_vertno)
            else: # if locations is a label
                patch_vertices.append([])

        # Create patch sources and save them as a group
        sources = []
        for (src_idx, vertno), patch_vertno, waveform, name, extent in zip(vertices, patch_vertices, data, names, extents):
            hemi = _extract_hemi(src[src_idx])
            sources.append(cls(
                name=name,
                src_idx=src_idx,
                vertno=vertno,
                patch_vertno=patch_vertno,
                waveform=waveform,
                sfreq=sfreq,
                hemi=hemi,
                extent=extent
            ))

        return sources


def _combine_sources_into_stc(sources, src):
    stc_combined = None
    
    for s in sources:
        stc_source = s.to_stc(src)
        if stc_combined is None:
            stc_combined = stc_source
            continue

        stc_combined = combine_stcs(stc_combined, stc_source)

    return stc_combined
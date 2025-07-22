"""
Classes that store all information about simulated sources.
Advantage of this approach over stc: in stc, we can only have one time series
per vertex, so if point sources coincide or patches overlap, we lose access
to the original time series.
"""

import numpy as np
import mne

from meegsim.utils import (
    vertices_to_mne,
    _extract_hemi,
    _get_center_of_mass,
    _get_param_from_stc,
    _hemi_to_index,
)


class _BaseSource:
    """
    An abstract class representing a source of activity.
    """

    kind = "base"

    def __init__(self, src_idx, waveform, std=1.0, name=""):
        # Current constraints:
        #  - one source corresponds to one waveform
        #  - all vertices belong to the same source space (e.g., hemisphere)
        self.src_idx = src_idx
        self.waveform = waveform
        self.std = std
        self.name = name

    @property
    def data(self):
        raise NotImplementedError(
            "The .data property should be implemented in a subclass."
        )

    @property
    def vertices(self):
        raise NotImplementedError(
            "The .vertices property should be implemented in a subclass."
        )

    def _check_compatibility(self, src):
        """
        Checks that the source is can be added to the provided src.

        Parameters
        ----------
        src: mne.SourceSpaces
            The source space where the source should be considered.

        Raises
        ------
        ValueError
            If the source does not exist in the provided src.
        """

        if self.src_idx >= len(src):
            raise ValueError(
                f"The {self.kind} source cannot be added to the provided src. "
                f"The {self.kind} source was assigned to source space {self.src_idx}, "
                f"which is not present in the provided src object."
            )

        own_vertno = [self.vertno] if self.kind == "point" else self.vertno
        missing_vertno = set(own_vertno) - set(src[self.src_idx]["vertno"])
        if missing_vertno:
            report_missing = ", ".join([str(v) for v in missing_vertno])
            raise ValueError(
                f"The {self.kind} source cannot be added to the provided src. "
                f"The source space with index {self.src_idx} does not "
                f"contain the following vertices: {report_missing}"
            )

    def to_label(self, src):
        """
        Get an mne.Label object containing all vertices belonging to the current
        source.

        Parameters
        ----------
        src : SourceSpaces
            The source space where the source should be considered.

        Returns
        -------
        label : Label
            The constructed label.
        """
        self._check_compatibility(src)

        # Make sure that we can turn the source into a label
        if src[self.src_idx]["type"] != "surf":
            raise ValueError(
                "Only sources in surface source spaces can be converted into a label"
            )

        # NOTE: we need to sort vertices before constructing the label
        vertno = np.atleast_1d(np.sort(self.vertices[:, 1]))
        return mne.Label(
            vertices=vertno,
            pos=src[self.src_idx]["rr"][vertno, :],
            # XXX: we should pass weights here once we implement patches with
            # amplitude decay
            values=np.ones_like(vertno),
            hemi="rh" if self.src_idx else "lh",
            name=self.name,
        )

    def to_stc(self, src, tstep, subject=None):
        """
        Convert the source into a SourceEstimate object in the context
        of the provided SourceSpaces.

        Parameters
        ----------
        src : SourceSpaces
            The source space where the source should be considered.
        tstep : float
            The sampling interval of the source time series (1 / sfreq).
        subject : str or None, optional
            Name of the subject that the stc corresponds to.
            If None, the subject name from the provided src is used if present.

        Returns
        -------
        stc : SourceEstimate
            SourceEstimate that corresponds to the source in the provided src.

        Raises
        ------
        ValueError
            If the source does not exist in the provided src.
        """

        self._check_compatibility(src)

        # Resolve the subject name as done in MNE
        if subject is None:
            subject = src[0].get("subject_his_id", None)

        # Convert the vertices to MNE format and construct the stc
        vertices = vertices_to_mne(self.vertices, src)
        return mne.SourceEstimate(
            data=self.data, vertices=vertices, tmin=0, tstep=tstep, subject=subject
        )


class PointSource(_BaseSource):
    """
    Point source of activity that is located in one of the vertices in
    the source space.

    Attributes
    ----------
    name : str
        The name of source.
    src_idx : int
        The index of source space that the point source belong to.
    vertno : int
        The vertex that the point source correspond to
    waveform : array
        The waveform of source activity.
    std : float, optional
        The standard deviation of the source activity (1 by default).
    hemi : str or None, optional
        Human-readable name of the hemisphere (e.g, lh or rh).
    """

    kind = "point"

    def __init__(self, name, src_idx, vertno, waveform, std=1.0, hemi=None):
        super().__init__(src_idx, waveform, std, name)

        self.vertno = vertno
        self.hemi = hemi

    def __repr__(self):
        # Use human readable names of hemispheres if possible
        src_desc = self.hemi if self.hemi else f"src[{self.src_idx}]"
        return f"<PointSource | {self.name} | {src_desc} | {self.vertno}>"

    @property
    def data(self):
        return np.atleast_2d(self.waveform)

    @property
    def vertices(self):
        return np.atleast_2d(np.array([self.src_idx, self.vertno]))

    @classmethod
    def _create(
        cls, src, times, n_sources, location, waveform, stds, names, random_state=None
    ):
        """
        This function creates point sources according to the provided input.
        """

        # Get the list of vertices (directly from the provided input or through the function)
        vertices = (
            location(src, random_state=random_state) if callable(location) else location
        )
        if len(vertices) != n_sources:
            raise ValueError("The number of sources in location does not match")

        # Get the corresponding number of time series
        data = (
            waveform(n_sources, times, random_state=random_state)
            if callable(waveform)
            else waveform
        )
        if data.shape[0] != n_sources:
            raise ValueError("The number of sources in waveform does not match")
        if data.shape[1] != len(times):
            raise ValueError("The number of samples in waveform does not match")

        # Get the std values if an stc was provided
        if isinstance(stds, mne.SourceEstimate):
            stds = _get_param_from_stc(stds, vertices)

        # Create point sources and save them as a group
        sources = []
        for (src_idx, vertno), waveform, std, name in zip(vertices, data, stds, names):
            hemi = _extract_hemi(src[src_idx])
            sources.append(
                cls(
                    name=name,
                    src_idx=src_idx,
                    vertno=vertno,
                    waveform=waveform,
                    std=std,
                    hemi=hemi,
                )
            )

        return sources


class PatchSource(_BaseSource):
    """
    Patch source of activity that is located in one of the vertices in
    the source space.

    Attributes
    ----------
    name : str
        The name of source.
    src_idx : int
        The index of source space that the patch source belong to.
    vertno : list
        The vertices that the patch sources correspond to including the central vertex.
    waveform : np.array
        The waveform of source activity.
    std : float, optional
        The standard deviation of the source activity (1 by default).
    hemi : str or None, optional
        Human-readable name of the hemisphere (e.g, lh or rh).
    """

    kind = "patch"

    def __init__(self, name, src_idx, vertno, waveform, std=1.0, hemi=None):
        super().__init__(src_idx, waveform, std, name)

        self.vertno = vertno
        self.hemi = hemi

    def __repr__(self):
        # Use human readable names of hemispheres if possible
        src_desc = self.hemi if self.hemi else f"src[{self.src_idx}]"
        n_vertno = len(self.vertno)
        vertno_desc = f"{n_vertno} vertex" if n_vertno == 1 else f"{n_vertno} vertices"
        return f"<PatchSource | {self.name} | {src_desc} | {vertno_desc} >"

    @property
    def data(self):
        # NOTE: the scaling factor is introduced to make the total variance of
        # patch activity invariant to the number of vertices in the patch
        scaling_factor = 1 / np.sqrt(len(self.vertno))
        return np.tile(self.waveform, (len(self.vertno), 1)) * scaling_factor

    @property
    def vertices(self):
        return np.array([[self.src_idx, v] for v in self.vertno])

    @classmethod
    def _create(
        cls,
        src,
        times,
        n_sources,
        location,
        waveform,
        stds,
        names,
        extents,
        subject,
        subjects_dir,
        random_state=None,
    ):
        """
        This function creates patch sources according to the provided input.
        """

        # Get the list of vertices (directly from the provided input or through the function)
        vertices = (
            location(src, random_state=random_state) if callable(location) else location
        )
        if len(vertices) != n_sources:
            raise ValueError("The number of sources in location does not match")

        # Get the corresponding number of time series
        data = (
            waveform(n_sources, times, random_state=random_state)
            if callable(waveform)
            else waveform
        )
        if data.shape[0] != n_sources:
            raise ValueError("The number of sources in waveform does not match")
        if data.shape[1] != len(times):
            raise ValueError("The number of samples in waveform does not match")

        # Pick subject name from src if not provided explicitly
        if subject is None:
            subject = src[0].get("subject_his_id", None)

        # Find patch vertices
        patch_vertices = []
        patch_stds = [] if isinstance(stds, mne.SourceEstimate) else stds
        for isource, extent in enumerate(extents):
            src_idx, vertno = vertices[isource]
            vertno = vertno if isinstance(vertno, list) else [vertno]

            # Get the std values if an stc was provided
            # NOTE: for now, we always use the value that corresponds to the
            # center of the patch but we could allow the user to select another
            # vertex
            if isinstance(stds, mne.SourceEstimate):
                center_vertno = _get_center_of_mass(src, src_idx, vertno)
                std = _get_param_from_stc(stds, [(src_idx, center_vertno)])
                patch_stds.append(std)

            # Add vertices as they are if no extent provided
            if extent is None:
                patch_vertices.append(vertno)
                continue

            # Grow the patch from center otherwise
            patch = mne.grow_labels(
                subject=subject,
                seeds=vertno,
                extents=extent,
                hemis=src_idx,
                subjects_dir=subjects_dir,
            )[0]

            # Prune vertices
            patch_vertno = [
                vert for vert in patch.vertices if vert in src[src_idx]["vertno"]
            ]
            patch_vertices.append(patch_vertno)

        # Create patch sources and save them as a group
        sources = []
        for (src_idx, _), patch_vertno, waveform, std, name in zip(
            vertices, patch_vertices, data, patch_stds, names
        ):
            hemi = _extract_hemi(src[src_idx])
            sources.append(
                cls(
                    name=name,
                    src_idx=src_idx,
                    vertno=patch_vertno,
                    waveform=waveform,
                    std=std,
                    hemi=hemi,
                )
            )

        return sources


def _combine_sources_into_stc(sources, src, tstep):
    """
    Create an stc object that contains the waveforms of all provided sources.

    Parameters
    ----------
    sources: list
        The list of point or patch sources.
    src: mne.SourceSpaces
        The source space with all candidate source locations.
    tstep: float
        The sampling interval of the source time series (1 / sfreq).

    Returns
    -------
    stc: mne.SourceEstimate
        The resulting stc object that contains all sources.
    """

    # Return immediately if no sources were provided
    if not sources:
        return None

    # Collect the data and vertices from all sources first
    data = []
    vertices = []
    for s in sources:
        s._check_compatibility(src)
        data.append(s.data)
        vertices.append(s.vertices)

    # Stack the data and vertices of all sources
    data_stacked = np.vstack(data)
    vertices_stacked = np.vstack(vertices)

    # Resolve potential repetitions: if several signals apply to the same
    # vertex, they should be summed
    unique_vertices, indices = np.unique(vertices_stacked, axis=0, return_inverse=True)
    n_unique = unique_vertices.shape[0]
    n_samples = data_stacked.shape[1]

    # Place the time courses correctly accounting for repetitions
    data = np.zeros((n_unique, n_samples))
    for idx_orig, idx_new in enumerate(indices):
        data[idx_new, :] += data_stacked[idx_orig, :]

    # Convert vertices to the MNE format
    vertices = vertices_to_mne(unique_vertices, src)

    return mne.SourceEstimate(data, vertices, tmin=0, tstep=tstep)


def _get_point_sources_in_hemi(sources, hemi):
    """
    Collect the indices of vertices (vertno) for all point sources
    belonging to the provided hemisphere.

    Parameters
    ----------
    sources: list
        A list of sources that were added to the configuration.
    hemi: str
        Hemisphere (lh or rh).

    Returns
    -------
    vertno: list
        List of indices of the corresponding vertices.
    """
    src_idx = _hemi_to_index(hemi)
    return [
        s.vertno for s in sources if isinstance(s, PointSource) and s.src_idx == src_idx
    ]


def _get_patch_sources_in_hemis(sources, src, hemis):
    """
    Collect the vertices for all patch sources belonging to the provided
    hemisphere(s).

    Parameters
    ----------
    sources: list
        A list of sources that were added to the configuration.
    src: SourceSpaces
        The source space that contains all candidate locations.
    hemis: list
        The list of hemispheres to consider.

    Returns
    -------
    stc: SourceEstimate
        An stc object that contains 1 for every vertex that is included at least
        in one of the patches and a small value near zero for all other sources.
        Non-zero value is required for the subsequent stc.plot() call.
    """
    src_indices = [_hemi_to_index(hemi) for hemi in hemis]
    n_vertno = [len(s["vertno"]) for s in src]
    # NOTE: we use a small non-zero value here to avoid problems with the
    # calculation of colorbar range in mne.viz.Brain. It is chosen to be smaller
    # than the transparency threshold (0.5) so that visually it is not noticeable
    data = [np.full((n,), 0.01) for n in n_vertno]
    for s in sources:
        if not isinstance(s, PatchSource) or s.src_idx not in src_indices:
            continue

        indices = np.searchsorted(src[s.src_idx]["vertno"], s.vertno)
        data[s.src_idx][indices] = 1
    data = np.hstack(data)

    return mne.SourceEstimate(
        data=data, vertices=[s["vertno"] for s in src], tmin=0.0, tstep=1.0
    )

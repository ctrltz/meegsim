import logging
import numpy as np
import warnings

from mne.io.constants import FIFF


logger = logging.getLogger("meegsim")


def combine_stcs(stc1, stc2):
    """
    Combines the data two SourceEstimate objects. If a vertex is present in both
    stcs (e.g., as a source of 1/f noise in one and oscillation in the other),
    the corresponding signals are summed.

    Parameters
    ----------
    stc1: SourceEstimate
        First object.

    stc2: SourceEstimate
        Second object.

    Returns
    -------
    stc: SourceEstimate
        The resulting stc that contains all vertices and data from stc1 and stc2.
        If a vertex is present in both stcs, the corresponding signals are summed.
    """

    # Accumulate positions in stc1.data where time series from stc2.data
    # should be inserted
    inserters = list()

    # Keep track of the offset in stc.data while iterating over hemispheres
    offsets_old = [0]
    offsets_new = [0]

    stc = stc1.copy()
    new_data = stc2.data.copy()
    for vi, (v_old, v_new) in enumerate(zip(stc.vertices, stc2.vertices)):
        v_common, ind1, ind2 = np.intersect1d(v_old, v_new, return_indices=True)
        if v_common.size > 0:
            # Sum up signals for vertices common to stc1 and stc2
            ind1 = ind1 + offsets_old[-1]
            ind2 = ind2 + offsets_new[-1]
            stc.data[ind1] += new_data[ind2]

            # Delete the common vertices from stc2 since they do not need
            # to be processed anymore
            new_data = np.delete(new_data, ind2, axis=0)
            v_new = v_new[np.isin(v_new, v_common, invert=True)]

        # Find where to insert the remaining vertices from stc2
        inds = np.searchsorted(v_old, v_new)
        stc.vertices[vi] = np.insert(v_old, inds, v_new)
        inserters += [inds.copy()]
        offsets_old += [len(v_old)]
        offsets_new += [len(v_new)]

    inds = [ii + offset for ii, offset in zip(inserters, offsets_old[:-1])]
    inds = np.concatenate(inds)
    stc.data = np.insert(stc.data, inds, new_data, axis=0)

    return stc


def normalize_variance(data):
    """
    Divide the time series by their standard deviation to make their
    variance equal to 1.

    Parameters
    ----------
    data: array, shape (n_series, n_samples)
        Time series to be normalized.

    Returns
    -------
    data_norm: array
        Normalized time series. The variance of each row is equal to 1.
    """
    # NOTE: make a copy to keep the original waveform intact
    data_norm = data.copy()
    if data_norm.ndim == 1:
        return data_norm / np.std(data_norm)

    data_norm /= np.std(data_norm, axis=-1)[:, np.newaxis]
    return data_norm


def _extract_hemi(src):
    """
    Extract a human-readable name (lh or rh) for the provided source space
    if it is a surface one.

    Parameters
    ----------
    src: dict
        The source space to process. It should be one of the elements stored
        in the mne.SourceSpaces structure.

    Returns
    -------
    hemi: str or None
        'lh' and 'rh' are returned for left and right hemisphere, respectively.
        None is returned otherwise.
    """

    if "type" not in src or "id" not in src:
        raise ValueError(
            "The provided source space does not have the mandatory "
            "internal fields ('id' or 'type'). Please check the code "
            "that was used to generate and/or manipulate the src. "
            "It should not change or remove these fields."
        )

    if src["type"] != "surf":
        return None

    if src["id"] == FIFF.FIFFV_MNE_SURF_LEFT_HEMI:
        return "lh"

    if src["id"] == FIFF.FIFFV_MNE_SURF_RIGHT_HEMI:
        return "rh"

    raise ValueError(
        "Unexpected ID for the provided surface source space. "
        "Please check the code that was used to generate and/or "
        "manipulate the src, it should not change the 'id' field."
    )


def get_sfreq(times):
    """
    Calculate the sampling frequency of a sequence of time points.

    Parameters
    ----------
    times: ndarray
        A sequence of time points assumed to be uniformly spaced.

    Returns
    -------
    out : float
        The sampling frequency
    """

    # Check if the number of time points is less than 2
    if len(times) < 2:
        raise ValueError("The times array must contain at least two points.")

    # Calculate the differences between consecutive time points
    dt = np.diff(times)

    # Check if the mean difference is different from the first difference
    if not np.isclose(np.mean(dt), dt[0]):
        raise ValueError("Time points are not uniformly spaced.")

    return 1 / dt[0]


def unpack_vertices(vertices_lists):
    """
    Unpack a list of lists of vertices into a list of tuples.

    Parameters
    ----------
    vertices_lists : list of lists
        A list where each element is a list of vertices correspond to
        different source spaces (one or two).

    Returns
    -------
    list of tuples
        A list of tuples, where each tuple contains:
        - index: The index of the source space.
        - vertno: Vertices in corresponding source space.
    """

    if isinstance(vertices_lists, list) and not all(
        isinstance(vertices, list) for vertices in vertices_lists
    ):
        warnings.warn(
            "Input is not a list of lists. Will be assumed that there is one source space.",
            UserWarning,
        )
        vertices_lists = [vertices_lists]

    unpacked_vertices = []
    for index, vertices in enumerate(vertices_lists):
        for vertno in vertices:
            unpacked_vertices.append((index, vertno))
    return unpacked_vertices


def vertices_to_mne(vertices, src):
    """
    Convert the vertices to the MNE format (list of lists).
    """

    vertices = np.array(vertices)
    packed_vertices = [[] for _ in src]
    for src_idx in np.unique(vertices[:, 0]):
        src_vertices = vertices[vertices[:, 0] == src_idx, :]
        src_vertno = list(np.sort(src_vertices[:, 1]))
        packed_vertices[src_idx] = src_vertno

    return packed_vertices


def _hemi_to_index(hemi):
    """
    Get the index of the hemisphere (0 for lh, 1 for rh).
    """
    return ["lh", "rh"].index(hemi)


def _get_param_from_stc(stc, vertices):
    """
    Extract parameter values for specified vertices from the provided stc.

    Parameters
    ----------
    stc : mne.SourceEstimate
        The stc object that contains values for all vertices.
    vertices: list
        List of tuples (src_idx, vertno) corresponding to the vertices of interest.

    Returns
    -------
    values : array
        One value from stc for each vertex.
    """
    values = np.zeros((len(vertices),))

    # NOTE: we only support surface source estimates for now
    offsets = [0, len(stc.vertices[0])]
    for i, (src_idx, vertno) in enumerate(vertices):
        idx = offsets[src_idx] + np.searchsorted(stc.vertices[src_idx], vertno)
        values[i] = stc.data[idx]

    return values


def _get_center_of_mass(src, src_idx, vertno):
    """
    For a given set of vertices, select one that is the closest to the
    center of mass in terms of Euclidean distance.

    Parameters
    ----------
    src : SourceSpaces
        The source spaces with information about all vertices.
    src_idx : int
        The index of source space to be considered.
    vertno : list
        The list of vertex indices to be considered.

    Returns
    -------
    center_vertno : int
        The vertex from the provided list that is the closest to the center of mass.
    """
    pos = src[src_idx]["rr"][vertno, :]
    mean_pos = pos.mean(axis=0)
    dist = np.sum((pos - mean_pos) ** 2, axis=1)

    return vertno[np.argmin(dist)]

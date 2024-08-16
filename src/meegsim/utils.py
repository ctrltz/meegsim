import logging
import numpy as np
import warnings

from mne.io.constants import FIFF
from scipy.special import i1, i0


logger = logging.getLogger('meegsim')


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


def normalize_power(data):
    """
    Divide the time series by its norm to normalize the variance.

    Parameters
    ----------
    data: array, shape (n_series, n_samples)
        Time series to be normalized.

    Returns
    -------
    data: array
        Normalized time series. The norm of each row is equal to 1.
    """

    data /= np.linalg.norm(data, axis=1)[:, np.newaxis]
    return data

  
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

    if 'type' not in src or 'id' not in src:
        raise ValueError("The provided source space does not have the mandatory "
                         "internal fields ('id' or 'type'). Please check the code "
                         "that was used to generate and/or manipulate the src. "
                         "It should not change or remove these fields.")

    if src['type'] != 'surf':
        return None
    
    if src['id'] == FIFF.FIFFV_MNE_SURF_LEFT_HEMI:
        return 'lh'
    
    if src['id'] == FIFF.FIFFV_MNE_SURF_RIGHT_HEMI:
        return 'rh'
    
    raise ValueError(f"Unexpected ID for the provided surface source space. "
                     f"Please check the code that was used to generate and/or "
                     f"manipulate the src, it should not change the 'id' field.")


def get_sfreq(times):
    """
    Calculate the sampling frequency of a sequence of time points.

    Parameters:
    ----------
    times: ndarray
        A sequence of time points assumed to be uniformly spaced.

    Returns:
    ----------
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

    if isinstance(vertices_lists, list) and not all(isinstance(vertices, list) for vertices in vertices_lists):
        warnings.warn("Input is not a list of lists. Will be assumed that there is one source space.", UserWarning)
        vertices_lists = [vertices_lists]

    unpacked_vertices = []
    for index, vertices in enumerate(vertices_lists):
        for vertno in vertices:
            unpacked_vertices.append((index, vertno))
    return unpacked_vertices


def theoretical_plv(kappa):
    return i1(kappa) / i0(kappa)

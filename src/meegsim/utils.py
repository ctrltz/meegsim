import numpy as np    
    
    
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

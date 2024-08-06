import numpy as np

from mne._fiff.constants import FIFF

def combine_sources_into_stc(sources, src, sfreq):
    stc_combined = None
    
    for s in sources:
        stc_source = s.to_stc(src, sfreq)
        if stc_combined is None:
            stc_combined = stc_source
            continue

        stc_combined = combine_stcs(stc_combined, stc_source)

    return stc_combined    
    
    
def combine_stcs(stc1, stc2):
    """
    Extension of stc.expand to work with data from another stc
    """
    inserters = list()
    offsets_old = [0]
    offsets_new = [0]
    
    stc = stc1.copy()
    new_data = stc2.data.copy()
    for vi, (v_old, v_new) in enumerate(zip(stc.vertices, stc2.vertices)):
        # Sum up signals for vertices common to stc1 and stc2
        v_common, ind1, ind2 = np.intersect1d(v_old, v_new, return_indices=True)
        if v_common.size > 0:
            ind1 = ind1 + offsets_old[-1]
            ind2 = ind2 + offsets_new[-1]
            stc.data[ind1] += new_data[ind2]
            new_data = np.delete(new_data, ind2, axis=0)
            v_new = v_new[np.isin(v_new, v_common, invert=True)]

        # Add vertices that are specific to s2
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

    """
    data /= np.sqrt(np.linalg.norm(data, axis=1))[:, np.newaxis]
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

# def src_vertno_to_vertices(src, src_idx, vertno):
#     n_vertno = [len(s['vertno']) for s in src]
#     offset = sum(n_vertno[:src_idx])
#     index = np.where(src[src_idx]['vertno'] == vertno)[0][0]
#     return [offset + index]
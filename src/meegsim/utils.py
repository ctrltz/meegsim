import numpy as np


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


# def src_vertno_to_vertices(src, src_idx, vertno):
#     n_vertno = [len(s['vertno']) for s in src]
#     offset = sum(n_vertno[:src_idx])
#     index = np.where(src[src_idx]['vertno'] == vertno)[0][0]
#     return [offset + index]

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
    unpacked_vertices = []
    for index, vertices in enumerate(vertices_lists):
        for vertno in vertices:
            unpacked_vertices.append((index, vertno))
    return unpacked_vertices

def pack_vertices(unpacked_vertices):
    """
    Pack a list of tuples into a list of lists of vertices.

    Parameters
    ----------
    unpacked_vertices : list of tuples
        A list of tuples, where each tuple contains:
        - index: The index of the source space.
        - vertno: A vertex number in the corresponding source space.

    Returns
    -------
    list of lists
        A list where each element is a list of vertices corresponding to
        different source spaces.
    """
    packed_vertices = {}
    for index, vertno in unpacked_vertices:
        if index not in packed_vertices:
            packed_vertices[index] = []
        packed_vertices[index].append(vertno)

    # Convert the dictionary to a list of lists
    # Sort keys to maintain order and ensure consistent output
    return [packed_vertices[i] for i in sorted(packed_vertices.keys())]


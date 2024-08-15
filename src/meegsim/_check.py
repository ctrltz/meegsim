"""
This module contains all functions that check the provided input
for SourceSimulator:

 - location and waveform functions
 - SNR values
 - parameters for adjusting SNR
 - coupling parameters
 - source names
"""

import numpy as np

from collections.abc import Sized
from functools import partial

from .utils import logger


def check_callable(name, fun, *args, **kwargs):
    """
    Check whether the provided function can be run successfully.
    The function is always run with random_state set to 0 for consistency.

    Parameters
    ----------
    name: str
        The name of the function to describe the context in the error message.
    fun: callable
        The function to be checked.
    *args: tuple
        Positional arguments that need to be passed to the function.
    **kwargs: dict, optional
        Keyword arguments the need to be passed to the function.

    Returns
    -------
    result
        The result of the function if the call was successful.

    Raises
    ------
    Exception
        Any exception that occurs during the function call.
    """

    try:
        return fun(*args, **kwargs, random_state=0)
    except:
        logger.error(f'An error occurred when trying to call the '
                     f'provided {name} function')
        raise


def check_vertices_list_of_tuples(vertices):
    """
    Check that the provided vertices are a list of tuples corresponding
    to our format:

    [(src_idx, vertno), (src_idx, vertno), ...]

    Parameters
    ----------
    vertices
        The input to be checked.

    Raises
    ------
    ValueError
        If any violation of the format is detected.
    """

    if not isinstance(vertices, list | tuple):
        raise ValueError(f"Expected vertices to be a list or a tuple, "
                         f" got {type(vertices)}")
    
    for i, el in enumerate(vertices):
        if not isinstance(el, list | tuple):
            raise ValueError(f"Expected each element of the vertices list to "
                             f"be a list or a tuple, does not hold for "
                             f"element {el}")

        if len(el) != 2:
            raise ValueError(f"Expected each element of the vertices list to "
                             f"contain 2 values, does not hold for element {el}")


def check_vertices_in_src(vertices, src):
    """
    Checks that all vertices are present in the provided src.

    Parameters
    ----------
    vertices: list
        The vertices to be checked.
    src: mne.SourceSpaces
        The source space which should contain all vertices.

    Raises
    ------
    ValueError
        In case any vertex is not present in the provided src.
    """
    for i, v in enumerate(vertices):
        src_idx, vertno = v
        if src_idx >= len(src):
            raise ValueError(f"Vertex {v} belongs to the source space {src_idx}, "
                             f"which is not present in the provided src")
        
        if vertno not in src[src_idx]['vertno']:
            raise ValueError(f"Vertex {v} is not present in the provided src[{src_idx}]")


def check_location(location, location_params, src):
    """
    Check the user input for the location of sources.
    If location is a callable function, it should not lead to an error.
    The result of the location function (if callable) or the location itself
    (if a list) should be a list of tuples (src_idx, vertno), and all locations
    should be present in the src.

    Parameters
    ----------
    location: np.array or callable
        The user input for the location.
    location_params: dict, optional
        Additional parameters to the location function, if needed.
    src: mne.SourceSpaces
        A source space that is used to validate the location function.
    
    Returns
    -------
    location: np.array or callable
        Checked location array or function.
    n_vertices: the number of vertices
        The number of vertices that are created.

    Raises
    ------
    ValueError
        If any violation of the expected format is detected.
    Expection
        If any expection occurs during the function call.
    """

    vertices = location
    if callable(location):
        location = partial(location, **location_params)
        vertices = check_callable('location', location, src)

    check_vertices_list_of_tuples(vertices)
    check_vertices_in_src(vertices, src)

    return location, len(vertices)


def check_waveform(waveform, waveform_params, n_sources):
    """
    
    """
    data = waveform
    if callable(waveform):
        n_samples = 1000
        times = np.arange(n_samples) / n_samples
        waveform = partial(waveform, **waveform_params)
        data = check_callable('waveform', waveform, n_sources, times)
        
    if data.shape[0] != n_sources:
        raise ValueError(
            f"The number of sources in the provided array or in the result of"
            f"the provided function for source waveform does not match: "
            f"expected {n_sources}, got {data.shape[0]}"
        )
    
    if callable(waveform) and data.shape[1] != n_samples:
        raise ValueError(
            f"The number of samples in the result of the provided function"
            f"for source waveform does not match: expected {n_samples}, "
            f"got {data.shape[1]}"
        )
    
    return waveform


def check_names(names, group, n_sources, existing):
    # TODO: check the provided source names
    #  - names are unique, not empty and do not yet exist in the structure
    # Check the provided names, broadcast to match the number of vertices
    if not names:
        # Autogenerate the names if they were not generated
        # No checks are needed, return immediately 
        return [f'auto-{group}-s{idx}' for idx in range(n_sources)]
    
    # Check the number of the provided names
    if len(names) != n_sources:
        raise ValueError('The number of provided source names does not match '
                         'the number of sources')
    
    # TODO: Check that all names aren't empty and don't start with auto (reserved for us)
    # TODO: Check that all names are not taken yet

    return names


def check_snr(snr, n_vertices):
    if snr is not None:
        raise NotImplementedError('Adjustment of SNR is not supported yet')
        # TODO: check that the number of SNR values matches the number of vertices
        # or it is a single SNR value that can be applied to all vertices
    
    return snr


def check_snr_params(snr_params):
    # TODO: we could try to extract fmin and fmax from waveform_params but
    # not sure how confusing will it be, a dedicated waveform class could be
    # easier to understand
    pass


def check_coupling():
    # coupling_edge = list(coupling.keys())[0]
    # coupling_params = list(coupling.values())[0]
    # name1, name2 = coupling_edge[0]
    # if missing:
    #     raise ValueError(f"The configuration contains no sources with the following names: {', '.join(missing)}")
    # self.check_if_exist([name1, name2])
    pass
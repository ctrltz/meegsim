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

from functools import partial
import warnings

from .coupling import COUPLING_PARAMETERS
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

    if not isinstance(vertices, (list, tuple)):
        raise ValueError(f"Expected vertices to be a list or a tuple, "
                         f" got {type(vertices)}")
    
    for i, el in enumerate(vertices):
        if not isinstance(el, (list, tuple)):
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
    If location is a callable, it should not lead to an error.
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
    location: np.array or functools.partial
        Checked location list or function (partial object which does not 
        require additional arguments anymore).
    n_vertices: the number of vertices
        The number of vertices that are created.

    Raises
    ------
    ValueError
        If any violation of the expected format is detected.
    Exception
        If any exception occurs during the function call.
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
    Check the user input for the waveforms of source activity.
    If waveform is a callable, it should not lead to an error.
    The result of the waveform function (if callable) or the waveform itself
    (if an array) should have the number of rows equal to the number of sources.
    
    Parameters
    ----------
    waveform: array or callable
        User input for the waveform.
    waveform_params: dict, optional
        Additional arguments for the waveform function, if needed.
    n_sources: int
        Number of expected sources. This number should be derived based on the
        provided input for location.

    Returns
    -------
    waveform: np.array or functools.partial
        Checked waveform array or function (partial object which does not 
        require additional arguments anymore).

    Raises
    ------
    ValueError
        If any violation of the expected array shape is detected.
    Exception
        If any exception occurs during the function call.
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


def check_names(names, n_sources, existing):
    """
    Check the user input for source names.
    The number of names should match the number of sources to be defined.
    In addition, all names should be unique and non-empty strings, 
    which don't start with auto and are not already in the structure.

    Parameters
    ----------
    names: list
        The list of names to be added
    n_sources: int
        The number of sources to be added.
    existing: list
        The list of names which are already assigned to other sources.

    Raises
    ------
    ValueError
        If any of the aforementioned checks fail.
    """
     
    # Check the number of the provided names
    if len(names) != n_sources:
        raise ValueError('The number of provided source names does not match '
                         'the number of defined sources')

    # All names should be non-empty strings
    for name in names:
        if not isinstance(name, str):
            actual_type = type(name).__name__
            raise ValueError(f"Expected all names to be strings, got {actual_type}: {name}")
        
        if not name:
            raise ValueError(f"All names should not be empty")
        
        if name.startswith('auto'):
            raise ValueError(f"Name {name} should not start with auto, this prefix "
                             f"is reserved for autogenerated names")
        
        if name in existing:
            raise ValueError(f"Name {name} is already taken by another source")
        
    # Check that all names are unique
    if len(names) != len(set(names)):
        raise ValueError('All names should be unique')


def check_snr(snr, n_sources):
    """
    Check the user input for SNR: it can either be None (no adjustment of SNR),
    a single float value that applies to all sources or an array of values
    with one for each source.

    Parameters
    ----------
    snr: None, float, or array
        The provided value(s) for SNR
    n_sources: int
        The number of sources.
    
    Raises
    ------
    ValueError
        If the provided SNR value(s) do not follow the format described above.
    """

    if snr is None:
        return None
    
    snr = np.ravel(np.array(snr))
    if snr.size != 1 and snr.size != n_sources:
        raise ValueError(
            f'Expected either one SNR value that applies to all sources or '
            f'one SNR value for each of the {n_sources} sources, got {snr.size}'
        )
    
    # Only positive values make sense, raise error if negative ones are provided
    if np.any(snr < 0):
        raise ValueError('Each SNR value should be positive')

    # Broadcast to all sources if a single value was provided
    if snr.size == 1:
        snr = np.tile(snr, (n_sources,))
    
    return snr


def check_snr_params(snr_params, snr):
    """
    Check the user input for SNR parameters: if the SNR is adjusted (i.e., not None),
    then fmin and fmax should be present in the dictionary to define a frequency band.

    Parameters
    ----------
    snr_params: dict
        The provided dictionary with parameters of the SNR adjustment.
    snr: None, float, or array
        The provided value for SNR
    
    Raises
    ------
    ValueError
        If the provided snr_params dictionary does not have the necessary parameters.
    """
    if snr is None:
        return snr_params
    
    if 'fmin' not in snr_params or 'fmax' not in snr_params:
        raise ValueError(
            'Frequency band limits are required for the adjustment of SNR. '
            'Please add fmin and fmax to the snr_params dictionary.'
        )
    
    if snr_params['fmin'] < 0 or snr_params['fmax'] < 0:
        raise ValueError('Frequency limits should be positive')
    
    return snr_params


def check_if_source_exists(name, existing):
    """
    Check if a source exists when trying to set the coupling.
    
    Parameters
    ----------
    name: str
        The name of the source to be checked.
    existing: list of str
        The name of all existing sources

    Raises
    ------
    ValueError
        If the provided source name is not in the list of existing ones.
    """
    if name not in existing:
        raise ValueError(f'Source {name} was not defined yet')


def check_coupling_params(method, coupling_params, coupling_edge):
    """
    Check whether all required coupling parameters were provided for the
    selected method.
    
    Parameters
    ----------
    method: str
        The name of the coupling method.
    coupling_params: dict
        The coupling parameters for the selected method.
    coupling_edge: tuple
        The coupling edge that the provided parameters apply to.
        It is only used to be more specific in the error message.

    Raises
    ------
    ValueError
        If the provided dictionary does not contain all required parameters.
    """
    for param in COUPLING_PARAMETERS[method]:
        if param not in coupling_params:
            raise ValueError(
                f'The {param} parameter is required for the {method} method '
                f'but was not defined for the {coupling_edge} coupling edge.'
            )


def check_coupling(coupling_edge, coupling_params, common_params, names, existing):
    """
    Check whether the provided coupling edge and parameters are valid.
    
    Parameters
    ----------
    coupling_edge: tuple
        The coupling edge (source, target) that the provided parameters apply to.
    coupling_params: dict
        The coupling parameters that were defined for this edge specifically.
    common_params: dict
        The coupling parameters that were defined for all edges.
    names: list of str
        The names of sources that exist in the simulation.
    existing: dict
        The coupling that already was added to the simulation

    Raises
    ------
    ValueError
        If source or target do not exist in the simulation.
        If the coupling edge was defined previously.
        If the coupling method or any of the required parameters for the method
        are not provided.
    """
    # Check that both source names already exist
    source, target = coupling_edge
    check_if_source_exists(source, names)
    check_if_source_exists(target, names)

    # Check that this coupling edge has not been already added
    if coupling_edge in existing:
        raise ValueError(
            f'The coupling edge {coupling_edge} already exists in the '
            f'simulation, and multiple definitions are not allowed.'
        )
    
    # Add common parameters to the edge-specific ones
    params = coupling_params.copy()
    params.update(common_params)

    # Check that the coupling method was defined
    if 'method' not in params:
        raise ValueError(f'Coupling method was not defined for the edge {coupling_edge}')
    
    # Check that all required coupling parameters were specified for the selected method
    method = params['method']
    check_coupling_params(method, params, coupling_edge)
    
    return params


def check_extents(extents, n_sources):

    # check if extents is a list, otherwise make it a list
    if not isinstance(extents, list):
        extents = [extents]
    # if extent is single number, propagate it to all patch sources
    if len(extents) == 1:
        extents = extents * n_sources

    for extent in extents:
        if extent is not None:
            # Check if each extent is a number
            if not isinstance(extent, (int, float)):
                raise ValueError(f"Extent {extent} must be a number.")

            # Check if each extent is positive
            if extent <= 0:
                raise ValueError(f"Extent {extent} must be a positive number.")

            # Issue a warning if any extent exceeds 1000 mm
            if extent > 1000:
                warnings.warn(
                    f"The extent {extent} (radius in mm) is more than 1000 mm. "
                    "Are you sure that the patch is supposed to be that big?",
                    UserWarning)


    return extents

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
import matplotlib.pyplot as plt
import warnings

from functools import partial
from matplotlib.colors import is_color_like

from meegsim.utils import logger


VIZ_SOURCE_TYPES = ["point", "patch", "noise", "candidate"]


def check_numeric(context, value, bounds=(None, None), allow_none=True):
    if allow_none and value is None:
        return value

    try:
        value = float(value)
    except (TypeError, ValueError):  # None leads to TypeError
        raise ValueError(f"Expected {context} to be a float number, got {type(value)}")

    lo, hi = bounds
    if lo is not None and value < lo:
        raise ValueError(f"Expected {context} to be {lo} or greater")
    if hi is not None and value > hi:
        raise ValueError(f"Expected {context} to be {hi} or lower")

    return value


def check_callable(context, fun, *args, **kwargs):
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
        logger.error(
            f"An error occurred when trying to call the "
            f"provided function for: {context}"
        )
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
        raise ValueError(
            f"Expected vertices to be a list or a tuple, got {type(vertices)}"
        )

    for i, el in enumerate(vertices):
        if not isinstance(el, (list, tuple)):
            raise ValueError(
                f"Expected each element of the vertices list to "
                f"be a list or a tuple, does not hold for "
                f"element {el}"
            )

        if len(el) != 2:
            raise ValueError(
                f"Expected each element of the vertices list to "
                f"contain 2 values, does not hold for element {el}"
            )


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
    for v in vertices:
        src_idx, vertno = v
        if src_idx >= len(src):
            raise ValueError(
                f"Vertex {v} belongs to the source space {src_idx}, "
                f"which is not present in the provided src"
            )

        vertno = [vertno] if not isinstance(vertno, list) else vertno
        missing_vertno = set(vertno) - set(src[src_idx]["vertno"])
        if missing_vertno:
            report_missing = ", ".join([str(v) for v in missing_vertno])
            vertex_desc = "Vertex" if len(missing_vertno) == 1 else "Vertices"
            verb = "is" if len(missing_vertno) == 1 else "are"
            raise ValueError(
                f"{vertex_desc} {report_missing} {verb} not present in the provided "
                f"src[{src_idx}]"
            )


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
        vertices = check_callable("location", location, src)

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
        data = check_callable("waveform", waveform, n_sources, times)

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
        raise ValueError(
            "The number of provided source names does not match "
            "the number of defined sources"
        )

    # All names should be non-empty strings
    for name in names:
        if not isinstance(name, str):
            actual_type = type(name).__name__
            raise ValueError(
                f"Expected all names to be strings, got {actual_type}: {name}"
            )

        if not name:
            raise ValueError("All names should not be empty")

        if name.startswith("auto"):
            raise ValueError(
                f"Name {name} should not start with auto, this prefix "
                f"is reserved for autogenerated names"
            )

        if name in existing:
            raise ValueError(f"Name {name} is already taken by another source")

    # Check that all names are unique
    if len(names) != len(set(names)):
        raise ValueError("All names should be unique")


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
            f"Expected either one SNR value that applies to all sources or "
            f"one SNR value for each of the {n_sources} sources, got {snr.size}"
        )

    # Only positive values make sense, raise error if negative ones are provided
    if np.any(snr < 0):
        raise ValueError("Each SNR value should be positive")

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

    if "fmin" not in snr_params or "fmax" not in snr_params:
        raise ValueError(
            "Frequency band limits are required for the adjustment of SNR. "
            "Please add fmin and fmax to the snr_params dictionary."
        )

    if snr_params["fmin"] < 0 or snr_params["fmax"] < 0:
        raise ValueError("Frequency limits should be positive")

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
        raise ValueError(f"Source {name} was not defined yet")


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

    # Test on a 10 second segment of white noise
    sfreq = 100
    rng = np.random.default_rng(seed=0)
    waveform = rng.random((10 * sfreq,))

    # Temporarily remove 'method' from coupling_params
    test_params = coupling_params.copy()
    test_params.pop("method")

    check_callable(
        f"coupling method, edge {coupling_edge}", method, waveform, sfreq, **test_params
    )


def check_coupling(coupling_edge, coupling_params, common_params, names, current_graph):
    """
    Check whether the provided coupling edge and parameters are valid.

    Parameters
    ----------
    coupling_edge: tuple
        The coupling edge (source, target) that the provided parameters apply to.
    coupling_params: dict
        The coupling parameters that were defined for this edge specifically.
    common_params: dict
        The coupling parameters that apply to all edges.
    names: list of str
        The names of sources that exist in the simulation.
    current_graph: nx.Graph
        The coupling graph that was already defined in the simulation

    Raises
    ------
    ValueError
        If source or target do not exist in the simulation.
        If the coupling edge was defined previously.
        If the coupling method or any of the required parameters for the method
        are not provided.
    """

    # Check that the coupling edge is defined as a tuple of two elements
    if not isinstance(coupling_edge, tuple):
        raise ValueError(f"Coupling edges {coupling_edge} should be defined as a tuple")
    if len(coupling_edge) != 2:
        raise ValueError(
            f"Coupling edges should contain two elements (names of "
            f"the source and the target), got {coupling_edge}"
        )

    # Check that both source names already exist
    source, target = coupling_edge
    check_if_source_exists(source, names)
    check_if_source_exists(target, names)

    # Check that the edge is not a self-loop
    if source == target:
        raise ValueError(
            f"The coupling edge {coupling_edge} is a self-loop, and "
            f"only connections between distinct sources are allowed."
        )

    # Check that this coupling edge has not been already added
    if current_graph.has_edge(*coupling_edge):
        raise ValueError(
            f"The coupling edge {coupling_edge} already exists in the "
            f"simulation, and multiple definitions are not allowed."
        )

    # Coupling parameters should be provided in a dictionary
    if not isinstance(coupling_params, dict):
        actual_type = type(coupling_params).__name__
        raise ValueError(
            f"Coupling parameters should be provided as a dictionary, "
            f"got {actual_type} for edge {coupling_edge}"
        )

    # Warn the user if some parameters were defined both for specific edges
    # and as common ones
    double_definition = set(common_params.keys()) & set(coupling_params.keys())
    if double_definition:
        double_defined = ", ".join(double_definition)
        warnings.warn(
            f"Parameters {double_defined} have double definition for edge "
            f"{coupling_edge}. Edge-specific values have higher priority."
        )

    # Overwrite the common coupling parameters with edge-specific ones
    params = common_params.copy()
    params.update(coupling_params)

    # Check that the coupling method was defined
    if "method" not in params:
        raise ValueError(
            f"Coupling method was not defined for the edge {coupling_edge}"
        )
    method = params["method"]

    # Check that the coupling method is a callable
    if not callable(method):
        raise ValueError(
            f"Expected coupling method to be a callable, "
            f"got {type(method).__name__} for edge {coupling_edge}"
        )

    # Check that all required coupling parameters were specified for the selected method
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
            if not isinstance(extent, (int, float, np.integer, np.floating)):
                raise ValueError(f"Extent {extent} must be a number.")

            # Check if each extent is positive
            if extent <= 0:
                raise ValueError(f"Extent {extent} must be a positive number.")

            # Issue a warning if any extent exceeds 1000 mm
            if extent > 1000:
                warnings.warn(
                    f"The extent {extent} (radius in mm) is more than 1000 mm. "
                    "Are you sure that the patch is supposed to be that big?",
                    UserWarning,
                )

    return extents


def check_colors(colors):
    if colors is None:
        return

    for key, color in colors.items():
        if key not in VIZ_SOURCE_TYPES:
            raise ValueError(f"Unexpected source type: {key}")

        # We expect a colormap for patches (easier for now but later we can
        # consider constructing a colormap from the provided color)
        if key == "patch" and color not in plt.colormaps():
            raise ValueError(
                "Expected a valid matplotlib colormap for the patch sources"
            )

        # For all other source types, one color is expected
        if key != "patch" and not is_color_like(color):
            raise ValueError(f"Expected a valid matplotlib color for the {key} sources")


def check_scale_factors(scale_factors):
    if scale_factors is None:
        return

    for key, scale_factor in scale_factors.items():
        if key not in VIZ_SOURCE_TYPES:
            raise ValueError(f"Unexpected source type: {key}")

        check_numeric(
            f"scale_factor of {key} sources", scale_factor, (0, None), allow_none=False
        )

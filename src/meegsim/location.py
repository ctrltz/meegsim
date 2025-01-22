"""
Methods for selecting locations of the sources that accept the following arguments:
    * it would be easier to have all methods accept src as the first argument by default
    * ideally, random_state (to allow reproducibility, still need to test how it would work)

Many options are already covered by mne.simulation.select_source_in_label so we can reuse the functionality under the hood.
"""

import numpy as np

from meegsim.utils import unpack_vertices


def select_random(src, *, n=1, vertices=None, sort_output=False, random_state=None):
    """
    Randomly selects a specified number of vertices from a given source space.

    Parameters
    ----------
    src : SourceSpaces
        An instance of source spaces to select the vertices from.

    n : int, optional
        The number of random vertices to select. By default, one vertex is selected.

    vertices : None (default) or list
        A subset of vertices to choose from.
        If None (default), the function uses all vertices from the provided ``src``.

    sort_output : bool, optional
        Indicates if sorting is needed for the output. By default, the output is
        not sorted.

    random_state : None (default) or int
        Seed for the random number generator. If None (default), results will vary
        between function calls. Use a fixed value for reproducibility.

    Returns
    -------
    selected : list
        A list of tuples (index of the source space, vertno of the selected vertex).
    """
    rng = np.random.default_rng(seed=random_state)

    if len(src) not in [1, 2]:
        raise ValueError(
            "Src must contain either one (volume) or two (surface) source spaces."
        )

    src_unpacked = unpack_vertices([list(s["vertno"]) for s in src])

    vertices = unpack_vertices(vertices) if vertices else src_unpacked
    vertices_not_in_src = set(vertices) - set(src_unpacked)
    if vertices_not_in_src:
        raise ValueError("Some vertices are not contained in the src.")

    if n > len(vertices):
        raise ValueError("Number of vertices to select exceeds available vertices.")

    selected_vertno = rng.choice(vertices, size=n, replace=False)
    selected_vertno = list(map(tuple, selected_vertno))
    if sort_output:
        selected_vertno = sorted(selected_vertno)

    return selected_vertno

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
    src : mne.SourceSpaces
        An instance of source spaces

    n : int, optional
        The number of random vertices to select. default = 1.

    vertices : list of lists, optional
        Specific vertices to choose from. If not provided, the function uses all vertices
        from src. default = None.

    sort_output : bool
        Indicates if sorting is needed for the output. default = False

    random_state : int or None, optional
        Seed for the random number generator. If None, it will be drawn
        automatically, and results will vary between function calls. default = None.

    Returns
    -------
    list of tuples
        A list of tuples, where each tuple contains:
        - index: The index of the source space.
        - vertno: The selected vertice.

    """
    rng = np.random.default_rng(seed=random_state)

    if len(src) not in [1, 2]:
        raise ValueError("Src must contain either one (volume) or two (surface) source spaces.")

    src_unpacked = unpack_vertices([s['vertno'] for s in src])

    vertices = unpack_vertices(vertices) if vertices else src_unpacked
    vertices_not_in_src = set(vertices) - set(src_unpacked)
    if vertices_not_in_src:
        raise ValueError("Some vertices are not contained in the src.")

    if n > len(vertices):
        raise ValueError("Number of vertices to select exceeds available vertices.")

    selected_vertno = rng.choice(vertices, size=n, replace=False)
    if sort_output:
        selected_vertno = sorted(selected_vertno)        

    return [(vert[0], vert[1]) for vert in selected_vertno]

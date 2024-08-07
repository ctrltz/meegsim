"""
Methods for selecting locations of the sources that accept the following arguments:
    * it would be easier to have all methods accept src as the first argument by default
    * ideally, random_state (to allow reproducibility, still need to test how it would work)

Many options are already covered by mne.simulation.select_source_in_label so we can reuse the functionality under the hood.
"""
import numpy as np
import mne

def select_random(src, *, n=1, vertices=None, random_state=None):
    """
    Randomly selects a specified number of vertices from a given source space.

    Parameters
    ----------
    src : mne.SourceSpaces
        An instance of source spaces

    n : int, optional
        The number of random vertices to select. default = 1.

    vertices : ndarray, optional
        An array of specific vertices to choose from. If not provided, the function uses all vertices
        from src.

    random_state : int or None, optional
        Seed for the random number generator. If None, it will be drawn
        automatically, and results will vary between function calls. default = None.

    Returns
    -------
    tuple of lists (if there are two source spaces)
        A tuple containing two lists:
        - lh_vertno: A list of selected vertices that belong to the left hemisphere (lh).
        - rh_vertno: A list of selected vertices that belong to the right hemisphere (rh).
    or a list (if there is only one source space)
        - vertno: A list of selected vertices

    """
    rng = np.random.default_rng(seed=random_state)

    if len(src) == 2:
        if vertices is None:
            vertices = np.concatenate([src[0]['vertno'], src[1]['vertno']+src[1]['np']])
        else:
            if not np.all(np.isin(vertices, np.concatenate([src[0]['vertno'], src[1]['vertno']+src[1]['np']]))):
                raise ValueError("Some vertices are not contained in the src.")
        if n > len(vertices):
            raise ValueError("Number of vertices to select exceeds available vertices.")

        selected_vertno = np.sort(rng.choice(vertices, size=n, replace=False))
        lh_vertno = [vert for vert in selected_vertno if vert <= src[1]['np']]
        rh_vertno = [vert for vert in selected_vertno if vert > src[1]['np']]
        return lh_vertno, rh_vertno
    elif len(src) == 1:
        if vertices is None:
            vertices = src[0]['vertno']
        else:
            if not np.all(np.isin(vertices, src[0]['vertno'])):
                raise ValueError("Some vertices are not contained in the src.")
        if n > len(vertices):
            raise ValueError("Number of vertices to select exceeds available vertices.")

        vertno = np.sort(rng.choice(vertices, size=n, replace=False))
        return vertno
    else:
        raise ValueError("Src must contain either one (volume) or two (surface) source spaces.")

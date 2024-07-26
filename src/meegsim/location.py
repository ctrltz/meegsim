"""
Methods for selecting locations of the sources that accept the following arguments:
    * it would be easier to have all methods accept src as the first argument by default
    * ideally, random_state (to allow reproducibility, still need to test how it would work)

Many options are already covered by mne.simulation.select_source_in_label so we can reuse the functionality under the hood.
"""

def select_random(src, *, n=1, vertices=None, random_state=None):
    # If vertices is None, use all vertices from the provided src
    # Otherwise only consider the provided vertices
    # Q: probably should still check the intersection of vertices and SNR and raise a warning if some vertices are not in src
    pass
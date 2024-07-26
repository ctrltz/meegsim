"""
Methods for setting the coupling between two signals
All methods should accept the following arguments:
  * time series of signal 1
  * coupling parameters
  * ideally, random_state (to allow reproducibility, still need to test how it would work)
  
All methods should return the time series of signals 1 and 2.

Methods currently in mind:
  * shifted copy (optionally with noise to control coupling parameters - later)
  * phase phase coupling using von Mises distribution (within-frequency for now?)
"""

def shifted_copy():
    pass


def ppc_von_mises():
    pass


COUPLING_FUNCTIONS = {
    'shifted_copy': shifted_copy,
    'ppc_von_mises': ppc_von_mises,
}

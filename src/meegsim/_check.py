import numpy as np
import logging

from functools import partial

from .utils import logger


def check_callable(name, fun, *args, **kwargs):
    """
    Check whether the provided function could be run without errors
    """
    try:
        return fun(*args, **kwargs)
    except:
        logger.info(f'An error occurred when trying to call the '
                    f'provided {name} function')
        raise


def check_location(location, location_params, src):
    vertices = location
    if callable(location):
        location = partial(location, **location_params)
        vertices = check_callable('location', location, 
                                  src, random_state=0)

    # TODO: check the format of vertices

    return location, len(vertices)


def check_waveform(waveform, waveform_params, n_sources, n_samples=100):
    # TODO: check the format of waveform
    #  - callable: check that possible to call, check the shape, return partial
    #  - array_like: check the shape, return as is
    data = waveform
    if callable(waveform):
        times = np.arange(n_samples) / n_samples
        waveform = partial(waveform, **waveform_params)
        data = check_callable('waveform', waveform, 
                              n_sources, times,
                              random_state=0)
        
    if data.shape != (n_sources, n_samples):
        # TODO: split into two errors with more informative messages?
        raise ValueError('The provided array/function for source waveform does not match other simulation parameters')
    
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
    
    return snr


def check_snr_params(snr_params):
    pass


def check_coupling():
    # coupling_edge = list(coupling.keys())[0]
    # coupling_params = list(coupling.values())[0]
    # name1, name2 = coupling_edge[0]
    # if missing:
    #     raise ValueError(f"The configuration contains no sources with the following names: {', '.join(missing)}")
    # self.check_if_exist([name1, name2])
    pass
import numpy as np
import mne

from mne.io.constants import FIFF


def prepare_source_space(types, vertices):
    assert len(types) == len(vertices), \
        "The number of types and the number of lists of vertices should match"

    # Create a simple dummy data structure for testing purposes
    src = []
    for i, (src_type, src_vertno) in enumerate(zip(types, vertices)):
        n_verts = len(src_vertno)

        # Generate random positions and random normals
        rr = np.random.rand(n_verts, 3) * 100
        nn = np.random.rand(n_verts, 3)

        # Set src ID according to the documentation
        src_id = FIFF.FIFFV_MNE_SURF_UNKNOWN
        if src_type == 'surf':
            assert i in [0, 1], "Surface source spaces should always go first"
            src_id = FIFF.FIFFV_MNE_SURF_RIGHT_HEMI if i else FIFF.FIFFV_MNE_SURF_LEFT_HEMI

        # Explicitly set types to match src objects that are created by MNE
        src_dict = dict(
            vertno=np.array(src_vertno),
            rr=np.array(rr),
            nn=np.array(nn),
            inuse=np.ones(n_verts, dtype=int),  # All vertices in use
            nuse=int(n_verts),
            type=str(src_type),
            id=int(src_id),
            np=int(n_verts),
            subject_his_id='meegsim'
        )
        src.append(src_dict)

    return mne.SourceSpaces(src)


def prepare_forward(n_channels, n_sources, 
                    ch_names=None, ch_types=None, sfreq=250):

    assert n_sources % 2 == 0, "Only even number of sources is supported"

    # Create a dummy info structure
    if ch_names is None:
        ch_names = [f'EEG{i+1}' for i in range(n_channels)]
    if ch_types is None:
        ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Generate random source space data (e.g., forward operator)
    fwd_data = np.random.randn(n_channels, n_sources)

    # Create a dummy source space
    lh_vertno = np.arange(n_sources // 2)
    rh_vertno = np.arange(n_sources // 2)

    src = prepare_source_space(['surf', 'surf'], [lh_vertno, rh_vertno])

    # Generate random source positions
    source_rr = np.random.rand(n_sources, 3)

    # Generate random source orientations
    source_nn = np.random.randn(n_sources, 3)
    source_nn /= np.linalg.norm(source_nn, axis=1, keepdims=True)

    # Create a forward solution
    forward = {
        'sol': {'data': fwd_data},
        '_orig_sol': fwd_data,
        'sol_grad': None,
        'info': info,
        'source_ori': 1,
        'surf_ori': True,
        'nsource': n_sources,
        'nchan': n_channels,
        'coord_frame': 1,
        'src': src,
        'source_rr': source_rr,
        'source_nn': source_nn,
        '_orig_source_ori': 1
    }

    # Convert the dictionary to an mne.Forward object
    fwd = mne.Forward(**forward)

    return fwd

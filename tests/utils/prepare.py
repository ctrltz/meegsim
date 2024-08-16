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
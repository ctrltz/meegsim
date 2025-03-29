import numpy as np
import mne

from mne.io.constants import FIFF

from meegsim.sources import PointSource, PatchSource


def prepare_source_space(types, vertices):
    assert len(types) == len(
        vertices
    ), "The number of types and the number of lists of vertices should match"

    # Create a simple dummy data structure for testing purposes
    src = []
    for i, (src_type, src_vertno) in enumerate(zip(types, vertices)):
        assert src_type in ["surf", "vol", "discrete"]
        n_verts = len(src_vertno)

        # Generate random positions and random normals
        rr = np.random.rand(n_verts, 3) * 100
        nn = np.random.rand(n_verts, 3)
        nn /= np.linalg.norm(nn, axis=1, keepdims=True)

        # Set src ID according to the documentation
        src_id = FIFF.FIFFV_MNE_SURF_UNKNOWN
        if src_type == "surf":
            assert i in [0, 1], "Surface source spaces should always go first"
            src_id = (
                FIFF.FIFFV_MNE_SURF_RIGHT_HEMI if i else FIFF.FIFFV_MNE_SURF_LEFT_HEMI
            )

        # Explicitly set types to match src objects that are created by MNE
        # Dictionary fields are ordered to match the documentation page of
        # mne.SourceSpaces
        src_dict = dict(
            id=int(src_id),
            type=str(src_type),
            np=int(n_verts),
            coord_frame=FIFF.FIFFV_COORD_MRI,
            rr=np.array(rr),
            nn=np.array(nn),
            nuse=int(n_verts),
            inuse=np.ones(n_verts, dtype=int),  # All vertices are in use
            vertno=np.array(src_vertno),
            subject_his_id="meegsim",
        )
        src.append(src_dict)

    return mne.SourceSpaces(src)


def prepare_info(n_channels, ch_names=None, ch_types=None, sfreq=250):
    if ch_names is None:
        ch_names = [f"EEG{i+1}" for i in range(n_channels)]
    if ch_types is None:
        ch_types = ["eeg"] * n_channels
    return mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)


def prepare_forward(n_channels, n_sources, ch_names=None, ch_types=None, sfreq=250):
    assert n_sources % 2 == 0, "Only an even number of sources is supported"

    # Create a dummy info structure
    info = prepare_info(n_channels, ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)

    # Generate random source space data (e.g., forward operator)
    fwd_data = np.random.randn(n_channels, n_sources)

    # Create a dummy source space
    lh_vertno = np.arange(n_sources // 2)
    rh_vertno = np.arange(n_sources // 2)

    # Generate a corresponding source space
    src = prepare_source_space(["surf", "surf"], [lh_vertno, rh_vertno])
    source_rr = np.vstack([s["rr"] for s in src])
    source_nn = np.vstack([s["nn"] for s in src])
    assert source_rr.shape == (n_sources, 3)
    assert source_nn.shape == (n_sources, 3)

    # Create a forward solution
    # Dictionary fields are ordered to match the documentation page of
    # mne.Forward, mri_head_t is not included
    forward = {
        "source_ori": FIFF.FIFFV_MNE_FIXED_ORI,
        "coord_frame": FIFF.FIFFV_COORD_MRI,
        "nsource": n_sources,
        "nchan": n_channels,
        "sol": {"data": fwd_data, "row_names": info.ch_names},
        "info": info,
        "src": src,
        "source_rr": source_rr,
        "source_nn": source_nn,
        "surf_ori": FIFF.FIFFV_MNE_FIXED_ORI,
    }

    # Convert the dictionary to an mne.Forward object
    fwd = mne.Forward(**forward)

    return fwd


def prepare_point_source(name, src_idx=0, vertno=0, n_samples=100):
    waveform = np.ones((n_samples,))
    return PointSource(name, src_idx, vertno, waveform)


def prepare_patch_source(name, src_idx=0, vertno=[0, 1], n_samples=100):
    waveform = np.ones((n_samples,))
    return PatchSource(name, src_idx, vertno, waveform)


def prepare_source_estimate(data, vertices):
    vertices = [np.array(el) for el in vertices]
    return mne.SourceEstimate(np.array(data), vertices, tmin=0, tstep=0.01)


def prepare_sinusoid(f, sfreq, duration):
    times = np.arange(sfreq * duration) / sfreq
    return np.sin(2 * np.pi * f * times)

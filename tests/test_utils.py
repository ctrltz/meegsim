import mne
import pytest

from mne._fiff.constants import FIFF
from meegsim.utils import _extract_hemi


def test_extract_hemi():
    src = mne.SourceSpaces([
        {'type': 'surf', 'id': FIFF.FIFFV_MNE_SURF_LEFT_HEMI},
        {'type': 'surf', 'id': FIFF.FIFFV_MNE_SURF_RIGHT_HEMI},
        {'type': 'vol', 'id': FIFF.FIFFV_MNE_SURF_UNKNOWN},
        {'type': 'discrete', 'id': FIFF.FIFFV_MNE_SURF_UNKNOWN},
    ])
    expected_hemis = ['lh', 'rh', None, None]
    
    for s, hemi in zip(src, expected_hemis):
        assert _extract_hemi(s) == hemi, f"Failed for {s['type']}"
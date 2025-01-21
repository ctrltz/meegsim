import mne
import numpy as np

from meegsim.sources import PointSource, PatchSource


DEFAULT_COLORS = dict(
    point="green", patch="Oranges", noise="#000000", candidate="yellow"
)
DEFAULT_SIZES = dict(point=0.75, noise=0.3, candidate=0.05)
DEFAULT_PLOT_KWARGS = dict(
    background="w",
    cortex="low_contrast",
    colorbar=False,
    clim=dict(kind="value", lims=[0, 0.5, 1]),
    transparent=True,
)
HEMIS = ["lh", "rh"]


def _get_point_sources_in_hemi(sources, hemi):
    src_idx = HEMIS.index(hemi)
    return [
        s.vertno for s in sources if isinstance(s, PointSource) and s.src_idx == src_idx
    ]


def _get_patch_sources_in_hemis(sources, src, hemis):
    # Collect vertices belonging to patches
    src_indices = [HEMIS.index(hemi) for hemi in hemis]
    n_vertno = [len(s["vertno"]) for s in src]
    data = [np.zeros((n,)) for n in n_vertno]
    for s in sources:
        if not isinstance(s, PatchSource) or s.src_idx not in src_indices:
            continue

        indices = np.searchsorted(src[s.src_idx]["vertno"], s.vertno)
        data[s.src_idx][indices] = 1
    data = np.hstack(data)

    return mne.SourceEstimate(
        data=data, vertices=[s["vertno"] for s in src], tmin=0.0, tstep=1.0
    )


def plot_source_configuration(
    sc,
    subject,
    hemi="lh",
    colors=None,
    sizes=None,
    show_noise_sources=True,
    show_candidate_locations=False,
    **brain_kwargs,
):
    # Overwrite the default values with user input
    source_colors = DEFAULT_COLORS.copy()
    if colors is not None:
        source_colors.update(colors)

    source_sizes = DEFAULT_SIZES.copy()
    if sizes is not None:
        source_sizes.update(sizes)

    hemis = HEMIS if hemi in ["both", "split"] else [hemi]

    # NOTE: we start with plotting all patch sources as an stc object
    # to ensure that the Brain object is initialized correctly
    patch_data_stc = _get_patch_sources_in_hemis(sc._sources.values(), sc.src, hemis)

    kwargs = DEFAULT_PLOT_KWARGS.copy()
    kwargs.update(brain_kwargs)
    brain = patch_data_stc.plot(
        subject=subject, hemi=hemi, colormap=source_colors["patch"], **kwargs
    )

    # Point/noise sources and candidate locations are added via
    # add_foci that needs to be run for each hemisphere separately
    for hemi in hemis:
        # All candidate locations (resource-heavy, disabled by default)
        if show_candidate_locations:
            src_idx = HEMIS.index(hemi)
            candidate_locations = sc.src[src_idx]["vertno"]
            brain.add_foci(
                candidate_locations,
                coords_as_verts=True,
                hemi=hemi,
                color=source_colors["candidate"],
                scale_factor=source_sizes["candidate"],
            )

        # Noise sources
        if show_noise_sources:
            brain.add_foci(
                _get_point_sources_in_hemi(sc._noise_sources.values(), hemi),
                coords_as_verts=True,
                hemi=hemi,
                color=source_colors["noise"],
                scale_factor=source_sizes["noise"],
            )

        # Point sources (always shown)
        brain.add_foci(
            _get_point_sources_in_hemi(sc._sources.values(), hemi),
            coords_as_verts=True,
            hemi=hemi,
            color=source_colors["point"],
            scale_factor=source_sizes["point"],
        )

    return brain

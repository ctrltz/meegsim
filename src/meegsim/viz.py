import mne

from meegsim.sources import PointSource


DEFAULT_COLORS = dict(
    point="green", patch="orange", noise="#444444", candidate="#aaaaaa"
)
DEFAULT_SIZES = dict(point=0.75, noise=0.3, candidate=0.05)
HEMIS = ["lh", "rh"]


def _get_point_sources_in_hemi(sources, hemi):
    src_idx = HEMIS.index(hemi)
    return [
        s.vertno for s in sources if isinstance(s, PointSource) and s.src_idx == src_idx
    ]


# def _get_patch_sources(sources, src):
#     data = np.zeros((n_vertno,))
#     for s in sources:
#         if not isinstance(s, PatchSource):
#             continue

#         patch_data = np.isin(src[s.src_idx]['vertno'], s.vertno).astype(int)
#         hemi_mask =
#         data[mask] = 1

#     print(data.sum())
#     if data.sum() > 0:
#         return mne.SourceEstimate(data=data, vertices=

#     return None
# return [
#     mne.Label(vertices=s.vertno, hemi=hemi) for s in sources
#     if isinstance(s, PatchSource) and s.src_idx == src_idx
# ]


def plot_source_configuration(
    sc,
    subject,
    hemi,
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

    # Initialize the brain plot
    Brain = mne.viz.get_brain_class()
    brain = Brain(subject=subject, hemi=hemi, **brain_kwargs)
    hemis = HEMIS if hemi in ["both", "split"] else [hemi]

    for hemi in hemis:
        # TODO: Patch sources
        # patch_data = _get_patch_sources_in_hemi(sc._sources.values(), sc.src, hemi)
        # if patch_data is not None:
        #     brain.add_data(patch_data, hemi=hemi, fmin=0, fmid=0.5, fmax=1)

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

        # Point sources
        brain.add_foci(
            _get_point_sources_in_hemi(sc._sources.values(), hemi),
            coords_as_verts=True,
            hemi=hemi,
            color=source_colors["point"],
            scale_factor=source_sizes["point"],
        )

    return brain

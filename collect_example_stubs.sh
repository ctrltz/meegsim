# NOTE: brain plots cannot be rendered with ReadTheDocs, therefore
# we build the docs locally and add generated brain images in correct places
# to show in the online version

STATIC_THUMB=docs/_static/example_stubs/thumb
STATIC_IMAGES=docs/_static/example_stubs/images

# examples/basics/02_plot_brain_configuration.py
cp docs/auto_examples/basics/images/thumb/sphx_glr_02_plot_brain_configuration_thumb.png $STATIC_THUMB
cp docs/auto_examples/basics/images/sphx_glr_02_plot_brain_configuration_001.png $STATIC_IMAGES

# examples/building_blocks/04_plot_std.py
cp docs/auto_examples/building_blocks/images/thumb/sphx_glr_04_plot_brain_std_thumb.png $STATIC_THUMB
cp docs/auto_examples/building_blocks/images/sphx_glr_04_plot_brain_std_001.png $STATIC_IMAGES
cp docs/auto_examples/building_blocks/images/sphx_glr_04_plot_brain_std_002.png $STATIC_IMAGES

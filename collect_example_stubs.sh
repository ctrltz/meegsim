# NOTE: brain plots cannot be rendered with ReadTheDocs, therefore
# we build the docs locally and add generated brain images in correct places
# to show in the online version

STATIC_THUMB=docs/_static/example_stubs/thumb
STATIC_IMAGES=docs/_static/example_stubs/images

# examples/basics/plot_brain_configuration.py
cp docs/auto_examples/basics/images/thumb/sphx_glr_02_plot_brain_configuration_thumb.png $STATIC_THUMB
cp docs/auto_examples/basics/images/sphx_glr_02_plot_brain_configuration_001.png $STATIC_IMAGES

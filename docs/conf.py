# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import pyvista
import os
import sys

import meegsim

from intersphinx_registry import get_intersphinx_mapping


project = "meegsim"
copyright = "2024, MEEGsim contributors"
author = "MEEGsim contributors"
release = meegsim.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # builtin
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    # contrib
    "numpydoc",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Hide class name for methods in the per-page TOC
toc_object_entries_show_parents = "hide"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_show_sourcelink = False
html_theme_options = {
    "icon_links": [
        dict(
            name="GitHub",
            url="https://github.com/ctrltz/meegsim",
            icon="fa-brands fa-square-github fa-fw",
        ),
    ],
    # include class methods in the per-page TOC
    "show_toc_level": 2,
    # version switcher dropdown & stable version banner
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "show_version_warning_banner": True,
    "switcher": {
        "json_url": "https://meegsim.readthedocs.io/en/latest/_static/versions.json",
        "version_match": "dev" if "dev" in release else release,
    },
}


# Autodoc
sys.path.insert(0, os.path.abspath("../src"))


# Autosummary
autosummary_generate = True


# Linkcode
code_url = "https://github.com/ctrltz/meegsim/blob/"


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None
    filename = info["module"].replace(".", "/")

    if "dev" in meegsim.__version__:
        branch = "master"
    else:
        branch = f"v{meegsim.__version__}"

    return f"{code_url}{branch}/src/{filename}.py"


# Numpydoc
numpydoc_attributes_as_param_list = True
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    # Python
    "bool": ":ref:`bool <python:typebool>`",
    # MNE-Python
    "Forward": "mne.Forward",
    "Info": "mne.Info",
    "Label": "mne.Label",
    "Raw": "mne.io.Raw",
    "SourceSpaces": "mne.SourceSpaces",
    "SourceEstimate": "mne.SourceEstimate",
    # MEEGsim
    "SourceConfiguration": "meegsim.configuration.SourceConfiguration",
}
numpydoc_xref_ignore = {
    "type",
    "optional",
    "default",
    "or",
    "shape",
    "n_series",
    "n_times",
}


# Intersphinx
intersphinx_mapping = get_intersphinx_mapping(
    packages={"matplotlib", "mne", "numpy", "python"}
)


# BibTeX
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"


# Enabling PyVista scraper
pyvista.BUILDING_GALLERY = True
pyvista.OFF_SCREEN = False


# Sphinx Gallery
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "filename_pattern": "/(plot_|run_)",
    "gallery_dirs": "auto_examples",
    "image_scrapers": ("matplotlib", "pyvista"),
}

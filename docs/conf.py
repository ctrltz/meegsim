# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys

import meegsim

from intersphinx_registry import get_intersphinx_mapping


project = 'meegsim'
copyright = '2024, MEEGsim contributors'
author = 'MEEGsim contributors'
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
    "sphinx_copybutton",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_show_sourcelink = False
html_theme_options = {
    "icon_links": [
        dict(
            name="GitHub",
            url="https://github.com/ctrltz/meegsim",
            icon="fa-brands fa-square-github fa-fw",
        ),
    ]
}


# Autodoc
sys.path.insert(0, os.path.abspath('../src'))


# Autosummary
autosummary_generate = True


# Linkcode
code_url = "https://github.com/ctrltz/meegsim/blob/"

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')

    if 'dev' in meegsim.__version__:
        branch = "master"
    else:
        branch = f'v{meegsim.__version__}'

    return f"{code_url}{branch}/src/{filename}.py"


# Numpydoc
numpydoc_class_members_toctree = False


# Intersphinx
intersphinx_mapping = get_intersphinx_mapping(
    packages={"python"}
)

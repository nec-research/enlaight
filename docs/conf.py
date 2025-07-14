"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import importlib.metadata

release = importlib.metadata.version("enlaight")
if "post" in release:
    version = release[:5] + "-post"
else:
    version = release

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "enlAIght"
copyright = "2025, NLE"
author = "Sascha Saralajew"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

autosummary_generate = True

# work around to fix the problem that __all__ in the import messes around with SPhinx
autosummary_mock_imports = [
    "enlaight.models",
]

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "autosummary_ignore_module_all": True,
}
autodoc_typehints = "none"

typehints_fully_qualified = False

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "scipy": (
        "https://docs.scipy.org/doc/scipy/",
        None,
    ),  # https://github.com/pvlib/pvlib-python/issues/1130
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
    "torch": ("https://pytorch.org/docs/main", None),
    "torchvision": ("https://pytorch.org/vision/main/", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
}

nbsphinx_execute = "always"

doctest_global_setup = """
from enlaight.core import *
from enlaight.models import *
import torch
import numpy
"""

exclude_patterns = ["build", "_build"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:

source_suffix = [".rst"]

# RUG is blocking the agent; the following links are persistent and should work
linkcheck_ignore = [
    (
        "https://research.rug.nl/en/publications/"
        "regularization-in-matrix-relevance-learning"
    ),
    (
        "https://research.rug.nl/en/publications/"
        "prototype-based-models-in-machine-learning"
    ),
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_short_title = "enlAIght"

# Logo of the doc
html_logo = html_static_path[0] + "/" + "logo.png"
html_theme_options = {
    "logo_only": False,
    "display_version": True,
}

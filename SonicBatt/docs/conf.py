# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SonicBatt'
copyright = '2024, Elias Galiounas'
author = 'Elias Galiounas'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Support automatic documentation
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage", # Automatically check if functions are documented
    "sphinx.ext.mathjax",  # Allow support for algebra
    "sphinx.ext.viewcode", # Include the source code in documentation
    "numpydoc",            # Support NumPy style docstrings
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import stack_to_chunk

project = "stack-to-chunk"
project_copyright = "2024, David Stansby"
author = "David Stansby"
version = stack_to_chunk.__version__
release = stack_to_chunk.__version__
language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

default_role = "any"
nitpicky = True
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    "theme.css",
]
html_theme_options = {
    "logo": {
        "text": "stack-to-chunk",
    },
}

html_use_index = False
html_show_sourcelink = False
html_show_copyright = False
html_sidebars = {"**": ["sidebar-nav-bs", "sidebar-ethical-ads"]}

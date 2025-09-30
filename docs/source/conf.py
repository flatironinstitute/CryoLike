# Configuration file for the Sphinx documentation builder.

## NOTE: This is a workaround because autodoc needs to be able
# to import the package, but it can't download the package from
# pip before we publish it. When we're ready to publish, remove these
# lines and update requirements.txt in the docs directory.

import sys
from pathlib import Path
sys.path.insert(0, str(Path('..', '..', 'src').resolve()))

# -- Project information

project = 'CryoLike'
copyright = '2024, Flatiron Institute'
author = 'Wai Shing Tang'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

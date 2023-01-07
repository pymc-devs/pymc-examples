import os, sys
from pathlib import Path
from sphinx.application import Sphinx

# -- Project information -----------------------------------------------------
project = "PyMC"
copyright = "2022, PyMC Community"
author = "PyMC Community"

# -- General configuration ---------------------------------------------------

sys.path.insert(0, os.path.abspath("../sphinxext"))

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_nb",
    "ablog",
    "sphinx_design",
    "sphinxext.opengraph",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
    "sphinx_codeautolink",
    "notfound.extension",
    "sphinx_gallery.load_style",
    "thumbnail_extractor",
    "sphinxext.rediraffe",
    "sphinx_remove_toctrees",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "*import_posts*",
    "**/.ipynb_checkpoints/*",
    "extra_installs.md",
    "page_footer.md",
    "**/*.myst.md",
]


def hack_nbsphinx(app: Sphinx) -> None:
    from nbsphinx import (
        depart_gallery_html,
        doctree_resolved,
        GalleryNode,
        NbGallery,
        patched_toctree_resolve,
    )
    from sphinx.environment.adapters import toctree

    from glob import glob

    nb_paths = glob("*/*.ipynb")
    nbsphinx_thumbnails = {}
    for nb_path in nb_paths:
        png_file = os.path.join(
            "thumbnails", os.path.splitext(os.path.split(nb_path)[-1])[0] + ".png"
        )
        nb_path_rel = os.path.splitext(nb_path)[0]
        nbsphinx_thumbnails[nb_path_rel] = png_file

    def builder_inited(app: Sphinx):
        if not hasattr(app.env, "nbsphinx_thumbnails"):
            app.env.nbsphinx_thumbnails = {}

    def do_nothing(*node):
        pass

    app.add_config_value("nbsphinx_thumbnails", nbsphinx_thumbnails, rebuild="html")
    app.add_directive("nbgallery", NbGallery)
    app.add_node(
        GalleryNode,
        html=(do_nothing, depart_gallery_html),
        latex=(do_nothing, do_nothing),
        text=(do_nothing, do_nothing),
    )
    app.connect("builder-inited", builder_inited)
    app.connect("doctree-resolved", doctree_resolved)

    # Monkey-patch Sphinx TocTree adapter
    toctree.TocTree.resolve = patched_toctree_resolve


def remove_index(app):
    """
    This removes the index pages so rediraffe generates the redirect placeholder
    It needs to be present initially for the toctree as it defines the navbar.
    """

    index_file = Path(app.outdir) / "index.html"
    index_file.unlink()

    app.env.project.docnames -= {"index"}
    yield "", {}, "layout.html"


def setup(app: Sphinx):
    hack_nbsphinx(app)
    app.connect("html-collect-pages", remove_index, 100)


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# theme options
html_theme = "pymc_sphinx_theme"
html_theme_options = {
    "secondary_sidebar_items": ["postcard", "page-toc", "edit-this-page", "sourcelink", "donate"],
    "navbar_start": ["navbar-logo"],
    "logo": {
        "link": "https://www.pymc.io",
    },
}
version = os.environ.get("READTHEDOCS_VERSION", "")
version = version if "." in version else "main"
doi_code = os.environ.get("DOI_READTHEDOCS", "10.5281/zenodo.5654871")
html_context = {
    "github_url": "https://github.com",
    "github_user": "pymc-devs",
    "github_repo": "pymc-examples",
    "github_version": version,
    "doc_path": "examples/",
    "sandbox_repo": f"pymc-devs/pymc-sandbox/{version}",
    "doi_url": f"https://doi.org/{doi_code}",
    "doi_code": doi_code,
    "default_mode": "light",
}


html_favicon = "../_static/PyMC.ico"
html_logo = "../_static/PyMC.png"
html_title = "PyMC example gallery"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["../_static"]
html_extra_path = ["../_thumbnails"]
templates_path = ["../_templates"]
html_sidebars = {
    "**": [
        "sidebar-nav-bs.html",
        "postcard_categories.html",
        "tagcloud.html",
    ],
}

# ablog config
blog_baseurl = "https://docs.pymc.io/projects/examples/en/latest/"
blog_title = "PyMC Examples"
blog_path = "blog"
blog_authors = {
    "contributors": ("PyMC Contributors", "https://docs.pymc.io"),
}
blog_default_author = "contributors"
fontawesome_included = True
# post_redirect_refresh = 1
# post_auto_image = 1
# post_auto_excerpt = 2

notfound_urls_prefix = "/projects/examples/en/latest/"

# MyST config
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath", "substitution"]
citation_code = f"""
```bibtex
@incollection{{citekey,
  author    = "<notebook authors, see above>",
  title     = "<notebook title>",
  editor    = "PyMC Team",
  booktitle = "PyMC examples",
  doi       = "{doi_code}"
}}
```
"""


myst_substitutions = {
    "pip_dependencies": "{{ extra_dependencies }}",
    "conda_dependencies": "{{ extra_dependencies }}",
    "extra_install_notes": "",
    "citation_code": citation_code,
}
nb_execution_mode = "off"

rediraffe_redirects = {
    "index.md": "gallery.md",
}
remove_from_toctrees = [
    "BART/*",
    "case_studies/*",
    "causal_inference/*",
    "diagnostics_and_criticism/*",
    "gaussian_processes/*",
    "generalized_linear_models/*",
    "mixture_models/*",
    "ode_models/*",
    "howto/*",
    "samplers/*",
    "splines/*",
    "survival_analysis/*",
    "time_series/*",
    "variational_inference/*",
]

# bibtex config
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

# OpenGraph config
# use default readthedocs integration aka no config here

codeautolink_autodoc_inject = False
codeautolink_concat_default = True

# intersphinx mappings
intersphinx_mapping = {
    "arviz": ("https://python.arviz.org/en/latest/", None),
    "bambi": ("https://bambinos.github.io/bambi", None),
    "einstats": ("https://einstats.python.arviz.org/en/latest/", None),
    "mpl": ("https://matplotlib.org/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "pymc": ("https://www.pymc.io/projects/docs/en/stable/", None),
    "pytensor": ("https://pytensor.readthedocs.io/en/latest/", None),
    "pmx": ("https://www.pymc.io/projects/experimental/en/latest/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

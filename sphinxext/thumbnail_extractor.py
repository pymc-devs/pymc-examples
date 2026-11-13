"""
Sphinx plugin to run generate a gallery for notebooks

Modified from the seaborn project, which modified the mpld3 project.
"""

import base64
import json
import os
import re
import shutil
import subprocess
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib import image

from sphinx.util import logging as sphinx_logging

logger = sphinx_logging.getLogger(__name__)

DOC_SRC = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMG_LOC = os.path.join(os.path.dirname(DOC_SRC), "_static", "PyMC.png")

HEAD = """
PyMC Example Gallery
====================

.. meta::
   :description: Hundreds of runnable PyMC example notebooks: regression,
       hierarchical models, Gaussian processes, time series, causal
       inference, and more Bayesian modeling recipes.

A collection of runnable Jupyter notebooks showing how to solve real modeling
problems with PyMC, from simple regression to advanced Bayesian workflows.

.. toctree::
   :hidden:

   object_index/index

"""

SECTION_TEMPLATE = """
.. _{section_id}:

{section_title}
{underlines}

.. grid:: 1 2 3 3
   :gutter: 4

"""

ITEM_TEMPLATE = """
   .. grid-item-card:: {title}
      :img-top: {image}
      :link: {doc_reference}
      :link-type: {link_type}
      :shadow: none
"""

intro_nb = {
    "title": "General Overview",
    "image": "https://raw.githubusercontent.com/pymc-devs/brand/main/pymc/pymc_logos/PyMC_square.svg",
    "doc_reference": "pymc:pymc_overview",
    "link_type": "ref",
}

glm_nb = {
    "title": "Simple Linear Regression",
    "image": "../_thumbnails/core_notebooks/glm_linear.png",
    "doc_reference": "pymc:glm_linear",
    "link_type": "ref",
}

model_comparison_nb = {
    "title": "Model Comparison",
    "image": "../_thumbnails/core_notebooks/model_comparison.png",
    "doc_reference": "pymc:model_comparison",
    "link_type": "ref",
}

prior_pred_nb = {
    "title": "Prior and Posterior Predictive Checks",
    "image": "../_thumbnails/core_notebooks/posterior_predictive.png",
    "doc_reference": "pymc:posterior_predictive",
    "link_type": "ref",
}

dimensionality_nb = {
    "title": "Distribution Dimensionality",
    "image": "../_thumbnails/core_notebooks/dimensionality.png",
    "doc_reference": "pymc:dimensionality",
    "link_type": "ref",
}

pytensor_nb = {
    "title": "PyMC and PyTensor",
    "image": "../_thumbnails/core_notebooks/pytensor_pymc.png",
    "doc_reference": "pymc:pymc_pytensor",
    "link_type": "ref",
}

external_nbs = {
    "introductory": [intro_nb, glm_nb],
    "fundamentals": [dimensionality_nb, pytensor_nb],
    "howto": [prior_pred_nb, model_comparison_nb],
}

folder_title_map = {
    "introductory": "Introductory",
    "fundamentals": "Library Fundamentals",
    "howto": "How to",
    "generalized_linear_models": "Generalized Linear Models",
    "case_studies": "Case Studies",
    "causal_inference": "Causal Inference",
    "gaussian_processes": "Gaussian Processes",
    "time_series": "Time Series",
    "spatial": "Spatial Analysis",
    "diagnostics_and_criticism": "Diagnostics and Model Criticism",
    "bart": "Bayesian Additive Regression Trees",
    "mixture_models": "Mixture Models",
    "survival_analysis": "Survival Analysis",
    "ode_models": "ODE models",
    "samplers": "MCMC",
    "variational_inference": "Variational Inference",
    "statistical_rethinking_lectures": "Statistical Rethinking Lectures",
}

# Cell tag that picks the cell whose first image becomes the notebook's gallery thumbnail.
THUMBNAIL_TAG = "gallery-thumbnail"


def _cell_images(cell):
    return [
        output["data"]["image/png"]
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    ]


def markdown_title_to_rst(title):
    """Convert the inline code and math a MyST title uses into their reST equivalents."""
    # reST only recognizes inline markup that is set off from surrounding text, and an escaped
    # space is an invisible separator, so it keeps markup inside words such as CO$_2$ working.
    title = re.sub(r"(?<!`)`([^`]+)`(?!`)", r"\\ ``\1``\\ ", title)
    return re.sub(r"\$([^$]+)\$", r"\\ :math:`\1`\\ ", title)


def is_tracked_by_git(filepath):
    # Git indexes a symlink as one entry and reports nothing for paths that run through it, so
    # resolve first. A git hook exports GIT_DIR, and with GIT_DIR set git takes the current
    # directory, here the Sphinx source directory, as the top of the work tree, so every notebook
    # would fall outside it. Dropping it lets git find the repository.
    env = {
        key: value for key, value in os.environ.items() if key not in ("GIT_DIR", "GIT_WORK_TREE")
    }
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(Path(filepath).resolve())],
            capture_output=True,
            check=False,
            env=env,
        )
    except FileNotFoundError:
        # git not available, assume all files are valid
        return True
    else:
        return result.returncode == 0


def create_thumbnail(infile, width=275, height=275, cx=0.5, cy=0.5, border=4):
    """Overwrites `infile` with a new file of the given size"""
    im = image.imread(infile)
    rows, cols = im.shape[:2]
    size = min(rows, cols)
    if size == cols:
        xslice = slice(0, size)
        ymin = min(max(0, int(cy * rows - size // 2)), rows - size)
        yslice = slice(ymin, ymin + size)
    else:
        yslice = slice(0, size)
        xmin = min(max(0, int(cx * cols - size // 2)), cols - size)
        xslice = slice(xmin, xmin + size)
    thumb = im[yslice, xslice]
    thumb[:border, :, :3] = thumb[-border:, :, :3] = 0
    thumb[:, :border, :3] = thumb[:, -border:, :3] = 0

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    ax = fig.add_axes((0, 0, 1, 1), aspect="auto", frameon=False, xticks=[], yticks=[])
    ax.imshow(thumb, aspect="auto", resample=True, interpolation="bilinear")
    fig.savefig(infile, dpi=dpi)
    plt.close(fig)
    return fig


class NotebookGenerator:
    """Tools for generating an example page from a file"""

    def __init__(self, filename, target_dir, folder):
        self.folder = folder
        self.basename = os.path.basename(filename)
        self.stripped_name = os.path.splitext(self.basename)[0]
        self.image_dir = os.path.join(target_dir, "_thumbnails", folder)
        self.png_path = os.path.join(self.image_dir, f"{self.stripped_name}.png")
        with open(filename, encoding="utf-8") as fid:
            self.json_source = json.load(fid)
        self.default_image_loc = DEFAULT_IMG_LOC

    def extract_title(self):
        """Return the notebook's first level-one markdown heading, or its file stem."""
        for cell in self.json_source["cells"]:
            if cell["cell_type"] != "markdown":
                continue
            for line in "".join(cell["source"]).splitlines():
                if line.startswith("# "):
                    return markdown_title_to_rst(line[2:].strip())

        logger.warning(f"No title heading found in {self.basename}", type="thumbnail_extractor")
        return self.stripped_name

    def extract_preview_pic(self):
        """Return the first image of the cell tagged ``THUMBNAIL_TAG``, else the notebook's last."""
        cells = self.json_source["cells"]
        tagged_cell = next(
            (cell for cell in cells if THUMBNAIL_TAG in cell.get("metadata", {}).get("tags", [])),
            None,
        )
        if tagged_cell is not None:
            tagged_pics = _cell_images(tagged_cell)
            if tagged_pics:
                return base64.b64decode(tagged_pics[0])
            logger.warning(
                f"The cell tagged {THUMBNAIL_TAG!r} in {self.basename} has no image output, "
                "using the last image",
                type="thumbnail_extractor",
            )

        pics = [pic for cell in cells for pic in _cell_images(cell)]
        if pics:
            return base64.b64decode(pics[-1])
        return None

    def gen_previews(self):
        preview = self.extract_preview_pic()
        if preview is not None:
            with open(self.png_path, "wb") as buff:
                buff.write(preview)
        else:
            logger.warning(
                f"Didn't find any pictures in {self.basename}", type="thumbnail_extractor"
            )
            shutil.copy(self.default_image_loc, self.png_path)
        create_thumbnail(self.png_path)


def main(app):
    logger.info("Starting thumbnail extractor.")

    working_dir = os.getcwd()
    os.chdir(app.builder.srcdir)
    try:
        write_gallery()
    finally:
        os.chdir(working_dir)


def write_gallery():
    """Write gallery.rst and the thumbnails, relative to the Sphinx source directory."""
    file = [HEAD]

    for folder, title in folder_title_map.items():
        target_dir = os.path.join("..", "_thumbnails", folder)
        os.makedirs(target_dir, exist_ok=True)

        items = [
            ITEM_TEMPLATE.format(
                title=descr["title"],
                image=descr["image"],
                doc_reference=descr["doc_reference"],
                link_type=descr["link_type"],
            )
            for descr in external_nbs.get(folder, [])
        ]

        for nb_path in sorted(glob(f"{folder}/*.ipynb")):
            if not is_tracked_by_git(nb_path):
                logger.info(f"Skipping {nb_path}, not tracked by git", type="thumbnail_extractor")
                continue

            nbg = NotebookGenerator(nb_path, "..", folder)
            nbg.gen_previews()
            items.append(
                ITEM_TEMPLATE.format(
                    title=nbg.extract_title(),
                    image=nbg.png_path,
                    doc_reference=os.path.join(folder, nbg.stripped_name),
                    link_type="doc",
                )
            )

        # An empty grid directive is invalid, so a section with no cards is left out entirely.
        if not items:
            continue

        file.append(
            SECTION_TEMPLATE.format(
                section_title=title, section_id=folder, underlines="-" * len(title)
            )
        )
        file.extend(items)

    with open("gallery.rst", "w", encoding="utf-8") as f:
        f.write("\n".join(file))


def setup(app):
    app.connect("builder-inited", main)
    return {"parallel_read_safe": True, "parallel_write_safe": True}

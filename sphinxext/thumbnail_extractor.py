"""
Sphinx plugin to run generate a gallery for notebooks

Modified from the seaborn project, which modified the mpld3 project.
"""
import base64
import json
import os
import shutil
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib import image

import sphinx

logger = sphinx.util.logging.getLogger(__name__)

DOC_SRC = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMG_LOC = os.path.join(os.path.dirname(DOC_SRC), "_static", "PyMC.png")

HEAD = """
PyMC Example Gallery
====================

.. toctree::
   :hidden:

   object_index/index

Core notebooks
--------------

.. grid:: 1 2 3 3
   :gutter: 4

   .. grid-item-card:: Introductory Overview of PyMC
      :img-top: https://raw.githubusercontent.com/pymc-devs/brand/main/pymc/pymc_logos/PyMC_square.svg
      :link: pymc:pymc_overview
      :link-type: ref
      :shadow: none

   .. grid-item-card:: GLM: Linear regression
      :img-top: ../_thumbnails/core_notebooks/glm_linear.png
      :link: pymc:glm_linear
      :link-type: ref
      :shadow: none

   .. grid-item-card:: Model Comparison
      :img-top: ../_thumbnails/core_notebooks/model_comparison.png
      :link: pymc:model_comparison
      :link-type: ref
      :shadow: none

   .. grid-item-card:: Prior and Posterior Predictive Checks
      :img-top: ../_thumbnails/core_notebooks/posterior_predictive.png
      :link: pymc:posterior_predictive
      :link-type: ref
      :shadow: none

   .. grid-item-card:: Distribution Dimensionality
      :img-top: ../_thumbnails/core_notebooks/dimensionality.png
      :link: pymc:dimensionality
      :link-type: ref
      :shadow: none

   .. grid-item-card:: PyMC and PyTensor
      :img-top: ../_thumbnails/core_notebooks/pytensor_pymc.png
      :link: pymc:pymc_pytensor
      :link-type: ref
      :shadow: none

"""

SECTION_TEMPLATE = """
.. _{section_id}:

{section_title}
{underlines}

.. grid:: 1 2 3 3
   :gutter: 4

"""

ITEM_TEMPLATE = """
   .. grid-item-card:: :doc:`{doc_reference}`
      :img-top: {image}
      :link: {doc_reference}
      :link-type: doc
      :shadow: none
"""

folder_title_map = {
    "generalized_linear_models": "(Generalized) Linear and Hierarchical Linear Models",
    "case_studies": "Case Studies",
    "causal_inference": "Causal Inference",
    "diagnostics_and_criticism": "Diagnostics and Model Criticism",
    "gaussian_processes": "Gaussian Processes",
    "ode_models": "Inference in ODE models",
    "samplers": "MCMC",
    "mixture_models": "Mixture Models",
    "survival_analysis": "Survival Analysis",
    "time_series": "Time Series",
    "variational_inference": "Variational Inference",
    "howto": "How to",
}


def create_thumbnail(infile, width=275, height=275, cx=0.5, cy=0.5, border=4):
    """Overwrites `infile` with a new file of the given size"""
    im = image.imread(infile)
    rows, cols = im.shape[:2]
    size = min(rows, cols)
    if size == cols:
        xslice = slice(0, size)
        ymin = min(max(0, int(cx * rows - size // 2)), rows - size)
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

    ax = fig.add_axes([0, 0, 1, 1], aspect="auto", frameon=False, xticks=[], yticks=[])
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

    def extract_preview_pic(self):
        """By default, just uses the last image in the notebook."""
        pic = None
        for cell in self.json_source["cells"]:
            for output in cell.get("outputs", []):
                if "image/png" in output.get("data", []):
                    pic = output["data"]["image/png"]
        if pic is not None:
            return base64.b64decode(pic)
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

    file = [HEAD]

    for folder, title in folder_title_map.items():

        nb_paths = glob(f"{folder}/*.ipynb")
        file.append(
            SECTION_TEMPLATE.format(
                section_title=title, section_id=folder, underlines="-" * len(title)
            )
        )
        target_dir = os.path.join("..", "_thumbnails", folder)
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)

        for nb_path in nb_paths:
            nbg = NotebookGenerator(nb_path, "..", folder)
            nbg.gen_previews()
            file.append(
                ITEM_TEMPLATE.format(
                    doc_reference=os.path.join(folder, nbg.stripped_name), image=nbg.png_path
                )
            )

    with open("gallery.rst", "w", encoding="utf-8") as f:
        f.write("\n".join(file))

    os.chdir(working_dir)


def setup(app):
    app.connect("builder-inited", main)

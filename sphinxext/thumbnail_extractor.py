"""
Sphinx plugin to run generate a gallery for notebooks

Modified from the seaborn project, which modified the mpld3 project.
"""
import base64
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib import image

import sphinx

logger = sphinx.util.logging.getLogger(__name__)

DOC_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IMG_LOC = os.path.join(os.path.dirname(DOC_SRC), "pymc-examples/_static", "PyMC.png")


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

    def __init__(self, filename, target_dir):
        self.basename = os.path.basename(filename)
        stripped_name = os.path.splitext(self.basename)[0]
        self.image_dir = os.path.join(target_dir, "_thumbnails", "thumbnails")
        self.png_path = os.path.join(self.image_dir, f"{stripped_name}.png")
        with open(filename) as fid:
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
    from glob import glob

    nb_paths = glob("*/*.ipynb")

    for nb_path in nb_paths:
        nbg = NotebookGenerator(nb_path, "..")
        nbg.gen_previews()


def setup(app):
    app.connect("builder-inited", main)

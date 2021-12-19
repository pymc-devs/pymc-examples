"""Build script that will execute notebooks."""

import os
from pathlib import Path
from glob import glob

notebooks = sorted(glob("notebooks/**/*.ipynb"))

for nb in notebooks:
    # We first execute the notebook from top to bottom, replacing the original,
    # thereby guaranteeing that we only execute the notebook once
    # before conversion to other formats.
    #
    # Original idea from: https://stackoverflow.com/a/62266448/1274908
    # (can also add  --ExecutePreprocessor.allow_errors=True to the end of the cmd)
    os.system(
        f"jupyter nbconvert {nb} --to notebook --inplace --execute --ExecutePreprocessor.kernel_name='pymc-examples' --ExecutePreprocesser.timeout=600"
    )
    # os.system(f"jupyter nbconvert {nb} --to pdf")
    # os.system(f"jupyter nbconvert {nb} --to latex")

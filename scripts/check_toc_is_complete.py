"""
Check that given Jupyter notebooks all appear in the table of contents.

This is intended to be used as a pre-commit hook, see `.pre-commit-config.yaml`.
You can run it manually with `pre-commit run check-toc --all`.
"""

import argparse
import ast

from pathlib import Path

if __name__ == "__main__":
    toc_examples = (Path("examples") / "table_of_contents_examples.js").read_text()
    toc_tutorials = (Path("examples") / "table_of_contents_tutorials.js").read_text()
    toc_keys = {
        **ast.literal_eval(toc_examples[toc_examples.find("{") :]),
        **ast.literal_eval(toc_tutorials[toc_tutorials.find("{") :]),
    }.keys()
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        notebook_name = path.relative_to('examples').with_suffix('').as_posix()
        assert notebook_name in toc_keys, f"Notebook '{notebook_name}' not added to table of contents!"

"""
Automatically add tags to notebook based on which PyMC3 classes are used.

E.g. if a notebook contains a section like

    :::{post} 30 Aug, 2021
    :tags: glm, mcmc, exploratory analysis
    :category: beginner
    :::

in a markdown cell, and uses the class pymc3.Categorical, then this script
will change that part of the markdown cell to:

    :::{post} 30 Aug, 2021
    :tags: glm, mcmc, exploratory analysis, pymc3.Categorical
    :category: beginner
    :::

Example of how to run this:

    python scripts/add_tags.py examples/getting_started.ipynb

"""
import sys
from myst_parser.main import to_tokens, MdParserConfig
import subprocess
import os
import json
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    args = parser.parse_args(argv)

    for file in args.files:
        # Find which PyMC3 classes are used in the code.
        output = subprocess.run(
            [
                "nbqa",
                "scripts.find_pm_classes",
                file,
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        classes = {f"pymc3.{obj}" for obj in output.stdout.splitlines()}

        # Tokenize the notebook's markdown cells.
        with open(file, encoding="utf-8") as fd:
            content = fd.read()
        notebook = json.loads(content)
        markdown_cells = "\n".join(
            [
                "\n".join(cell["source"])
                for cell in notebook["cells"]
                if cell["cell_type"] == "markdown"
            ]
        )
        config = MdParserConfig(enable_extensions=["dollarmath", "colon_fence"])
        tokens = to_tokens(markdown_cells, config=config)

        # Find a ```{post} or :::{post} code block, and look for a line
        # starting with tags: or :tags:.
        tags = None
        for token in tokens:
            if token.tag == "code" and token.info.startswith("{post}"):
                for line in token.content.splitlines():
                    if line.startswith("tags: "):
                        line_start = "tags: "
                        original_line = line
                        tags = {tag.strip() for tag in line[len(line_start) :].split(",")}
                        break
                    elif line.startswith(":tags: "):
                        line_start = ":tags: "
                        original_line = line
                        tags = {tag.strip() for tag in line[len(line_start) :].split(",")}
                        break

        # If tags were found, then append any PyMC3 classes which might have
        # been missed.
        if tags is not None:
            new_tags = ", ".join(sorted(tags.union(classes)))
            new_line = f"{line_start}{new_tags}"
            content = content.replace(original_line, new_line)
            with open(file, "w", encoding="utf-8") as fd:
                fd.write(content)


if __name__ == "__main__":
    exit(main())

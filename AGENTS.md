# AGENTS.md

## Repository structure

- `.ipynb` is the source of truth. A pre-commit jupytext hook auto-generates `.myst.md` from `.ipynb` — edit only the `.ipynb`.
- Only `.ipynb` files are used by Sphinx; `.myst.md` is excluded via `exclude_patterns` in `examples/conf.py`.
- `requirements-docs.txt` lists Sphinx/theme dependencies for the ReadTheDocs build only.
- `sphinxext/thumbnail_extractor.py` is a custom Sphinx extension that extracts thumbnail images from notebook outputs.

## Do not commit

- Agent-generated artifacts, scratch notes, or temporary files (e.g. `.github/pr-summaries/`, draft summaries).

## ReadTheDocs (remote) builds

- `nb_execution_mode = "off"` in `examples/conf.py` — notebooks are never executed during RTD builds. Missing Python packages (pymc, arviz, etc.) cannot cause RTD failures.
- RTD "Unknown problem" failures with short duration (~140s vs normal ~350s) are transient infrastructure/pip-install failures. Retrigger with an empty commit or from the RTD dashboard before investigating further.

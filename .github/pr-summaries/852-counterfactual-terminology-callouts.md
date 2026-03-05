# PR: Add counterfactual terminology callouts to quasi-experimental notebooks

Closes #852

## Issue Summary

Four quasi-experimental notebooks in `examples/causal_inference/` correctly use "counterfactual" in the Rubin/potential outcomes sense, but readers familiar with Pearl's causal ladder may wonder whether L2/L3 confusion applies. A brief callout in each notebook clarifies this.

## Root Cause

The notebooks lacked explicit terminology clarification distinguishing Rubin-sense counterfactuals (unobserved potential outcomes) from Pearl's L3 unit-level counterfactuals (which require abduction of unit-specific exogenous terms).

## Solution

Added a `:::{admonition}` callout to each of the four notebooks, tailored to the specific quasi-experimental method, explaining:
1. The notebook uses "counterfactual" in the potential outcomes (Rubin) sense
2. How this differs from Pearl's L3 counterfactuals
3. Why the usage is appropriate for the method

Also fixed a misleading do-operator reference in `excess_deaths.ipynb`.

## Changes Made

- `examples/causal_inference/interrupted_time_series.ipynb` + `.myst.md`: Added terminology callout after first mention of "counterfactual" in the introduction
- `examples/causal_inference/difference_in_differences.ipynb` + `.myst.md`: Added terminology callout at end of Introduction section
- `examples/causal_inference/regression_discontinuity.ipynb` + `.myst.md`: Added terminology callout in "Counterfactual questions" section
- `examples/causal_inference/excess_deaths.ipynb` + `.myst.md`: Added terminology callout after introduction; removed misleading "famous do-operator" reference from strategy list
- `examples/references.bib`: Added three bibliography entries (Rubin 1974, Imbens & Rubin 2015, Pearl 2009 2nd ed.)

## Testing

- [x] All callouts use consistent structure and cross-reference the `counterfactuals_do_operator` notebook
- [x] Both `.ipynb` and `.myst.md` versions updated in sync
- [x] Bibliography entries added in correct alphabetical position
- [x] `excess_deaths.ipynb` do-operator reference corrected
- [x] Existing code and narrative otherwise preserved

## Notes

- All callouts cross-reference the `{ref}`counterfactuals_do_operator`` notebook for the full L2/L3 explanation. If PR #850 renames that notebook, the cross-reference label may need updating.
- References cited: Rubin (1974), Imbens & Rubin (2015), Pearl (2009, 2nd ed.)

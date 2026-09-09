# PR: Add notebook — Unit-level counterfactuals with abduction, action, and prediction

Closes #851

## Issue Summary

The PyMC examples gallery had no notebook demonstrating unit-level counterfactuals (rung 3 of Pearl's causal ladder). Users may assume `pm.do` is sufficient for individual-level "what if?" questions, when it actually answers a population-level question (rung 2).

## Root Cause

No existing example demonstrated the three-step counterfactual procedure (abduction, action, prediction) or explained the conceptual difference between interventional and counterfactual queries.

## Solution

New notebook `unit_level_counterfactuals.ipynb` implementing the encouragement design from Pearl, Glymour, and Jewell (2016), *Causal Inference in Statistics: A Primer*, §4.2.3–4.2.4.

The notebook:
- Opens with Joe's concrete question (examples-first pedagogy)
- Simulates data from a known SCM, fits it with PyMC, and verifies coefficient recovery
- Computes the population intervention E[Y|do(H=2)] ≈ 0.8 and shows it gives the wrong answer for Joe
- Explains the conceptual shift from "residuals as noise" to "U as individual property"
- Implements all three steps (abduction, action, prediction) per posterior draw
- Shows both comparison figure (intervention vs counterfactual) and counterfactual posterior
- Includes summary table mapping intervention vs counterfactual to Pearl's causal ladder
- Ends with bolded-key-term summary and domain-transfer reflection prompt

## Changes Made

- `examples/causal_inference/unit_level_counterfactuals.ipynb`: New notebook (41 cells)
- `examples/causal_inference/unit_level_counterfactuals.myst.md`: Auto-generated MyST companion

## Testing

- [x] Valid JSON notebook structure verified
- [x] All 41 cells present with correct types
- [x] MyST companion file generated via jupytext
- [x] Cross-reference to existing `interventional_distribution` notebook included
- [x] Citation key `pearl2016causal` exists in `examples/references.bib`

## Notes

- Notebook is not yet executed (no outputs). Execution requires a PyMC environment with graphviz.
- Figures follow figure-excellence conventions: `FIG_WIDTH`/`FIG_HEIGHT` constants, semantic colors (`COLOR_POPULATION`, `COLOR_JOE`), technical captions.
- Narrative follows educational-narrative skill: examples-first, progressive complexity, callouts for pedagogical functions, summary with bolded key terms, reflection prompt.

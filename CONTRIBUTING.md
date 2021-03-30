# Guidelines for Contributing
As part of the PyMC3 library documentation, the guidelines to contribute to
pymc-examples are based on [PyMC3 contributing guidelines](https://github.com/pymc-devs/pymc3/blob/master/CONTRIBUTING.md). Please refer there
for a detailed description of the Fork-PR contributing workflow, see "Steps" section,
and note that you'll need to update the repository URLs and branch names.

This document therefore covers only some specific guidelines specific to this repository, mainly,
an adapted version of the "Pull Request Checklist" and some extra guidelines for
efficient collaboration with Jupyter notebooks.

## Before submitting a pull request
The notebooks in pymc-examples are in the process of being updated and reexecuted.
The main progress tracker is [this GitHub project](https://github.com/pymc-devs/pymc-examples/projects/1).

### About the notebook tracker project
This project serves as both tracker and organizer of the work needed on each of the example notebooks in this repo.
Each notebook will have its own issue where we can point out things to fix and discuss them.
These issue tickets are placed on one of the columns in this project based on the state of the notebook:

* **To Do:** notebooks in this column are outdated, don't use ArviZ or InferenceData (or do so only partially), use deprecated pymc3 arguments or show other practices that should be updated and improved.
* **General updates:** notebooks in this column have pymc3 code up to date with v3, but don't use ArviZ (or do so only partially)
* **ArviZ:** notebooks in this column use ArviZ but still have bad examples of pymc3 usage.
* **Best practices:** notebooks in this column use ArviZ and pymc3 best practices. This column alone does not represent any extra updates, it is only the place for notebooks fulfilling the requirements to be in both "general updates" and "ArviZ".
* **v4:** notebooks in this column use ArviZ and have been updated and executed with pymc3 v4.

Therefore, all notebooks will be progressively updated along this path:

```
         / --> General updates -- \
To Do --<                          >--> Best Practices (--> v4)
         \ -->      ArviZ      -- /
```

Each pull request should update a single notebook 1-2 positions to the right.
Before starting a work on a pull request look at the tracker issue of the
notebook you are planning to edit to make sure it is not being updated by someone
else.

**Note on labels**: The labels on an issue will apply to the most immediate 1 position update.
One issue can be labeled "good first issue" for updating from "To Do" to "General updates", but
that does not mean that updating from "To Do"->"ArviZ" or
"General Updates"->"Best Practices" won't be challenging.
The same could be true the other way around.
In case of doubt, don't hesitate to ask and read over the notebook to
see what changes are expected.

If there are no comments and nobody is working on this notebook,
comment on the ticket to make it evident to others, we will assign
the issue to you as soon as possible.
If the comment if more than two weeks old and there are no signs of
activity, leave a comment for a maintainer to assign the issue to you.

We encourage you to submit a pull request as soon as possible after commenting
and being assigned the issue and
add `[WIP]` in the title to indicate Work in Progress.

### About PR timeline
You are free and encouraged to work at your own pace as long as you keep
track of your progress in the issues and PRs. There is no deadline nor
maximum time an active PR can be open.

There is a maximum time of 2 weeks for inactive PRs,
if there is no activity for two weeks,
we will close the PR and the issue will be unassigned.
We will try to ping you a few days before that happens,
but not being receiving such ping does not mean you won't get unassigned.

If you know you won't be able to work during two weeks but plan to
continue your work afterwards, let us know by commenting when you'll be able
to retake the work.
Alternatively, you can also contact your reviewers on [Discourse](https://discourse.pymc.io/)

As for review timeline, while you may get some reviews in a few hours or even some minutes
if we happen to be working on related things, _you should not expect that to be the norm_.
You should expect to receive review(s) for your PRs in 1-2 days. If two and a half days
after submitting you still have not received any comment, let us know (i.e. tag whoever
opened the issue you are addressing in a new PR comment. If at any point we were
overwhelmed by PRs and delay this timeline, we will comment on your PR with an estimate
of when you can expect a proper review.

### In the event of a conflict
In the event of two or more people working on the same issue,
the general precedence will go to the person who first commented in the issue.
If no comments it will go to the first person to submit a PR for review.
Each situation will differ though, and the core contributors will make the best judgement call if needed.

### If the issue ticket has someone assigned to it
If the issue is assigned then precedence goes to the assignee.
However if there has been no activity for 2 weeks from assignment date,
the ticket is open for all again and will be unassigned.

## Pull request checklist

We recommended that your contribution complies with the following guidelines before you submit a pull request:

*  Use the pull request title to describe the issue and mention the issue number in the pull request description. This will make sure a link back to the original issue is created. For example, use `Use ArviZ in sampler stats notebook` as a title and link to [#46](https://github.com/pymc-devs/pymc-examples/issues/46) in the description.
   * Please do not submit PRs that are not addressing an issue already present in the issue tracker.

*  Prefix the title of incomplete contributions with `[WIP]` (to indicate a work in progress). WIPs may be useful to (1) indicate you are working on something to avoid duplicated work, (2) request broad review of functionality or API, or (3) seek collaborators.

* Make sure to run the whole notebook sequentially on a fresh kernel. You can do that with the
  "Restart & Run All" option before saving.

* No `pre-commit` errors: see the [Jupyter Notebook style](https://github.com/pymc-devs/pymc3/wiki/PyMC3-Jupyter-Notebook-Style-Guide) (and [Python code style](https://github.com/pymc-devs/pymc3/wiki/PyMC3-Python-Code-Style)) page from our Wiki on how to install and run it.

* Indicate how are you aiming to update the notebook (i.e. what is the target end column in the tracker). The pull request template has a template for this.

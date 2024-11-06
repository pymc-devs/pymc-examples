# Guidelines for Contributing
As part of the PyMC3 library documentation, the guidelines to contribute to
pymc-examples are based on 
[PyMC3 contributing guidelines](https://github.com/pymc-devs/pymc3/blob/master/CONTRIBUTING.md). 
Please refer there for a detailed description of the Fork-PR contributing workflow, 
see "Steps" section,and note that you'll need to update the repository URLs and 
branch names.

This document therefore covers only some specific guidelines specific to this 
repository, mainly, an adapted version of the "Pull Request Checklist" and some
extra guidelines for efficient collaboration with Jupyter notebooks.

## Before submitting a pull request
The notebooks in `pymc-examples` are in the process of being updated and reexecuted.
The main progress tracker is 
[this GitHub project](https://github.com/pymc-devs/pymc-examples/projects/1).

### About the notebook tracker project
This project serves as both tracker and organizer of the work needed on each of 
the example notebooks in this repo.
Each notebook will have its own issue where we can point out things to fix and 
discuss them.
These issue tickets are placed on one of the columns in this project based on 
the state of the notebook:

* **To Do:** notebooks in this column are potentially outdated, run on v3, don't
  follow the style guide, don't use best practices when using PyMC...
* **Best practices (v3):** notebooks in this column use ArviZ and PyMC v3 best 
  practices.
* **v4 (auto)**: Notebooks in this column have been updated and reexecuted with 
  PyMC v4, but following v3 style and patterns, and are not taking advantage of
  the new features introduced in v4.
* **Book style**: Notebooks in this column have had their content, style and
  formatting updated to take advantage of all the pymc-examples website features,
  but still need work on the code side (either because they still use v3 or 
  because they use v4 but don't take advantage of its new features). There is a 
  [webinar](https://pymc-data-umbrella.xyz/en/latest/webinars/contributing_to_documentation/index.html)
  about the pymc examples repo and contributing to it, part of the PyMC Data 
  Umbrella series.
* **Done:** notebooks in this column use ArviZ and have been updated and 
  executed with PyMC v4.

Therefore, all notebooks will be progressively updated along this path:

```
                                 / -->   Book style    -- \
To Do --> Best Practices (v3) --<                          >--> Done
                                 \ -->   v4 (auto)     -- /
```

See https://github.com/pymc-devs/pymc-examples/wiki/Notebook-updates-overview 
for a more detailed description of what each of the statuses mean.

Each pull request should update a single notebook 1-2 positions to the right.
Before starting a work on a pull request look at the tracker issue of the
notebook you are planning to edit to make sure it is not being updated by someone
else.

**Note on labels**: The labels on an issue generally apply to potential code changes.
You should ignore them if you are looking at content/style updates only.

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
Alternatively, you can also contact your reviewers on 
[Discourse](https://discourse.pymc.io/)

As for review timeline, while you may get some reviews in a few hours or even 
some minutes if we happen to be working on related things, 
_you should not expect that to be the norm_.
You should expect to receive review(s) for your PRs in 1 - 2 days. If 2 1/2 days
after submitting you still have not received any comment, let us know 
(i.e. tag whoever opened the issue you are addressing in a new PR comment). 
If at any point we were overwhelmed by PRs and delay this timeline, we will 
comment on your PR with an estimate of when you can expect a proper review.

### In the event of a conflict
In the event of two or more people working on the same issue,
the general precedence will go to the person who first commented in the issue.
If no comments it will go to the first person to submit a PR for review.
Each situation will differ though, and the core contributors will make the best
judgement call if needed.

### If the issue ticket has someone assigned to it
If the issue is assigned then precedence goes to the assignee.
However if there has been no activity for 2 weeks from assignment date,
the ticket is open for all again and will be unassigned.

## Pull request checklist

We recommended that your contribution complies with the following guidelines 
before you submit a pull request:

* Use the pull request title to describe the issue and mention the issue number
  in the pull request description. This will make sure a link back to the original
  issue is created. For example, use `Use ArviZ in sampler stats notebook` as a
  title and link to [#46](https://github.com/pymc-devs/pymc-examples/issues/46)
  in the description.
  * Please do not submit PRs that are not addressing an issue already present
    in the issue tracker.
  * If you want to add a new notebook and no issue related to it is present yet,
    open one so we can discuss the best way to add the content to the repo. 
    We have an issue template for that.

* Prefix the title of incomplete contributions with `[WIP]` (to indicate a work
  in progress). WIPs may be useful to (1) indicate you are working on something 
  to avoid duplicated work, (2) request broad review of functionality or API, 
  or (3) seek collaborators.

* Make sure to run the whole notebook sequentially on a fresh kernel. You can do
  that with the "Restart & Run All" option before saving.

* No `pre-commit` errors: see the 
  [Jupyter Notebook style](https://github.com/pymc-devs/pymc3/wiki/PyMC3-Jupyter-Notebook-Style-Guide)
  (and [Python code style](https://github.com/pymc-devs/pymc3/wiki/PyMC3-Python-Code-Style)) 
  page from our Wiki on how to install and run it.

* Indicate how are you aiming to update the notebook (i.e. what is the target
  end column in the tracker). The pull request template has a template for this.

## Contributor guide
In order to work and run the example notebooks you need to install the packages
in `requirements-write.txt`. To see how the notebook looks rendered, you can 
follow the instructions in the following paragraph or open a PR to see the 
preview in **readthedocs**.

The markdown cells in the notebook can use MyST, a superset of CommonMark markdown.
See https://myst-parser.readthedocs.io/en/latest/ and 
https://myst-nb.readthedocs.io/en/latest/ for documentation on their features 
and syntax.

To generate the draft standalone notebook gallery, you need to have installed
all the packages in `requirements-docs.txt` and to run 
`sphinx-build examples/ _build -b html` from the repository home directory. 
After building, you can see the preview of the docs by opening `_build/index.html` 
file with your browser.

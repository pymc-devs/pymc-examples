:::{attention}
This notebook uses libraries that are not PyMC dependencies
and therefore need to be installed specifically to run this notebook.
Open the dropdown below for extra guidance.
:::

::::::{dropdown} Extra dependencies install instructions
:icon: package-dependencies
In order to run this notebook (either locally or on binder) you won't only
need a working PyMC installation with all optional dependencies,
but also to install some extra dependencies.
For advise on installing PyMC itself, please refer to {ref}`pymc:installation`

You can install these dependencies with your preferred package manager, we provide
as an example the pip and conda commands below.

> $ pip install {{ pip_dependencies }}

Note that if you want (or need) to install the packages from inside the notebook instead
of the command line, you can install
the packages by running a variation of the pip command:

> import sys
>
> !{sys.executable} -m pip install {{ pip_dependencies }}

You should not run `!pip install` as it might install the package in a different
environment and not be available from the Jupyter notebook even if installed.

Another alternative is using conda instead:

> $ conda install {{ conda_dependencies }}

when installing scientific python packages with conda,
we recommend using [conda forge](https://conda-forge.org/index.html)

{{ extra_install_notes }}
::::::

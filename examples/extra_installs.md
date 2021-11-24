:::{attention}
This notebook uses libraries that are not PyMC dependencies
and therefore need to be installed specifically to run this notebook
:::

::::::{dropdown} Extra dependencies install instructions
:icon: package-dependencies
In order to run this notebook (either locally or on binder) you won't only
need a working PyMC installation, but also to install some extra dependencies.

You can install these dependencies with your preferred package manager, we provide
as an example the pip and conda commands below.

> pip install {{ pip_dependencies }}

Note that if you are running the notebook from binder or colab, you can install
the packages by running a variation of the pip command:

> import sys
>
> !{sys.executable} -m pip install {{ pip_dependencies }}

You should not run `!pip install` as it might install the package in a different
environment and not be available from the jupyter notebook even if installed.

Another alternative is using conda instead:

> conda install {{ conda_dependencies }}

when installing scientific python packages with conda,
we recommend using [conda forge](https://conda-forge.org/index.html)

{{ extra_install_notes }}
::::::

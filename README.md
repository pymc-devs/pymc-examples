# Proposal: `pymc-experimental` repository

As PyMC continues to mature and expand its functionality to accomodate more domains of application, we increasingly see cutting-edge methodologies, highly specialized statistical distributions, and complex models appear. While this adds to the functinoality and relevance of the project, it can also introduce instability and impose a burden on testing and quality control. To help address this, a `pymc-experimental` respository could act as a home for new additions to PyMC, which may include unusual probability distribitions, advanced model fitting algorithms, or any code that may be inappropriate to include in the `pymc` repository, but may want to be made available to users.

If implemented thoughtfully, a `pymc-experimental` repository could act as the first step in the PyMC development pipeline, where all novel code is introduced until it is obvious that it belongs in the main repository. This would improve the stability and streamline the testing overhead of the `pymc` respository.

## Questions

### What belongs in `pymc-experimental`?

- newly-implemented statistical methodologies
- distributions that are tricky to sample from or test
- infrequently-used fitting methods or distributions
- any code that requires additional optimization before it can be used in practice


### What does not belong in `pymc-experimental`?


### Should `pymc-experimental` be a submodule?


### Should there be more than one add-on repository?

Since there is a lot of code that we may not want in the main repository, does it make sense to have more than one additional repository? For exmaple, `pymc-experimental` may just include methods that are not fully developed, tested and trusted, while code that is known to work well and has adequate test coverage could reside in a `pymc-extras` (or similar) repository.


### How can we minimize the additional burden of additional project repositories?


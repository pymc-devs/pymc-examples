__version__ = "0.0.1"

import logging

_log = logging.getLogger("pmx")

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)


from pymc_experimental import distributions, gp, utils
from pymc_experimental.bart import *

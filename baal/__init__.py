from __future__ import absolute_import

from . import structures
from . import induce
from . import parse
from . import semantics


from .hacks import nnp_transformer, cc_transformer

import os
PATH = os.path.dirname(os.path.realpath(__file__))

OMNI = utils.DefaultSettings()
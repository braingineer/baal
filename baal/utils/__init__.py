from __future__ import absolute_import

from . import general
from . import timer
from . import loggers
from . import config

from .exceptions import *
from .general import *

def DefaultSettings():
    return config.Settings.default()

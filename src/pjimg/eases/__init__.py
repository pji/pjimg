"""
eases
~~~~~~~

Easing functions for image and video data.

.. automodule:: pjimg.eases.easess
.. automodule:: pjimg.eases.decorators

"""
import pjimg.eases.eases_ as eases_
from pjimg.eases.decorators import will_scale
from pjimg.eases.eases_ import *
from pjimg.eases.model import Ease
from pjimg.util import get_prefixed_functions as _get_prefixed_functions


# Mapping of registered eases.
eases = _get_prefixed_functions('ease_', eases_)

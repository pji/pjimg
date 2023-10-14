"""
imgease
~~~~~~~

Easing functions for image and video data.

.. automodule:: pjimg.imgease.imgeases
.. automodule:: pjimg.imgease.decorators

"""
import pjimg.imgease.imgeases as imgeases
from pjimg.imgease.decorators import will_scale
from pjimg.imgease.imgeases import *
from pjimg.imgease.model import Ease
from pjimg.util import get_prefixed_functions as _get_prefixed_functions


# Mapping of registered eases.
eases = _get_prefixed_functions('ease_', imgeases)

"""
imgblend
~~~~~~~~

Blending operations to use when combining two sets of image data.

Many of these are taken from:

    *   http://www.deepskycolors.com/archive/2010/04/21/
        formulas-for-Photoshop-blending-modes.html
    *   http://www.simplefilter.de/en/basics/mixmods.html


.. automodule:: pjimg.imgblend.ops
.. automodule:: pjimg.imgblend.decorators
"""
import pjimg.imgblend.ops as ops
from pjimg.imgblend.decorators import *
from pjimg.imgblend.model import Blend
from pjimg.imgblend.ops import *
from pjimg.util import get_prefixed_functions as _get_prefixed_functions


# Mapping of registered eases.
blends = _get_prefixed_functions('', ops)

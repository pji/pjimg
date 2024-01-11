"""
test_util
~~~~~~~~~

Unit tests for :mod:`pjimg.util.util`.
"""
import tests.spam as spam
from pjimg.util import util as u


# Test cases.
def test_get_free_rotation_size_2d():
    """Given the final size of an image, :func:`get_free_rotation_size_2d`
    should return the size of image data needed to allow that image to
    be rotated to any angle without the corners of the image going
    outside of the generated image data.
    """
    size = (1, 720, 1280)
    assert u.get_free_rotation_size_2d(size) == (1, 1469, 1469)


def test_get_free_rotation_size_2d_pivot_offset():
    """Given the final size of an image, :func:`get_free_rotation_size_2d`
    should return the size of image data needed to allow that image to
    be rotated to any angle without the corners of the image going
    outside of the generated image data. If given a pivot offset, the
    size should be calculated based on the offsetting the center of
    rotation by the given amount.
    """
    size = (1, 360, 640)
    loc = (0, 180, 320)
    assert u.get_free_rotation_size_2d(size, loc) == (1, 1469, 1469)


def test_get_prefixed_functions():
    """When given a prefix and an object, :func:`get_prefixed_functions`
    should return a :class:`dict` of functions within that object's
    namespace that start with the prefix.
    """
    assert u.get_prefixed_functions('spam_', spam) == {
        'eggs': spam.spam_eggs,
        'bacon': spam.spam_bacon,
        'baked_beans': spam.spam_baked_beans,
    }

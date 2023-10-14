"""
test_util
~~~~~~~~~

Unit tests for :mod:`pjimg.util.util`.
"""
import tests.spam as spam
from pjimg.util import util as u


# Test cases.
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

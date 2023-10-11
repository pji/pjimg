"""
test_imggen
~~~~~~~~~~~

Unit tests for :mod:`pjimg.imggen.model`.
"""
import pytest as pt

from pjimg.imggen import model as m


# Fixtures for Serializable.
@pt.fixture
def serial():
    """An object of a :class:`Serializable` class."""
    class Serial(m.Serializable):
        def __init__(self, spam, eggs):
            self.spam = spam
            self.eggs = eggs

    return Serial(1, 2)


# Tests for Serializable.
def test_Serializable_asdict(serial):
    """The :meth:`Serializable.asdict` method should return the
    class's public attributes as a dictionary that can be used as
    keyword arguments to initialize a new instance of the class.
    """
    assert serial.asdict() == {
        'spam': 1,
        'eggs': 2,
    }


def test_Serializeable_asargs(serial):
    """The :meth:`Serializable.asargs` method should return the
    class's public attributes as a tuple that can be used as
    positional arguments to initialize a new instance of the class.
    """
    assert serial.asargs() == (1, 2)


def test_Serializable_comparisons():
    """Two :class:`Serializable` objects should be able to be compared."""
    class Serial(m.Serializable):
        def __init__(self, spam, eggs):
            self.spam = spam
            self.eggs = eggs

    assert Serial(1, 2) == Serial(1, 2)
    assert Serial(1, 2) != Serial(1, 3)


def test_Serializable_repr(serial):
    """:class:`Serializable` objects should return a string useful for
    troubleshooting when coerced into a string.
    """
    assert repr(serial) == 'Serial(spam=1, eggs=2)'
    serial.spam = ''.join(str(n) for n in range(60))
    assert repr(serial) == "Serial(spam='0123...9', eggs=2)"
    serial.spam = b'3'
    assert repr(serial) == "Serial(spam=b'3', eggs=2)"


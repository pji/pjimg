"""
fixtures
~~~~~~~~

Common fixtures for pjimg.test_sources.
"""
import pytest as pt


@pt.fixture
def noise_attr_defaults():
    """Default attribute values for pjimg.sources.Noise subclasses."""
    return {
        'device': '',
        'seed': None,
    }


@pt.fixture
def noise_attr_set():
    """"Changed attribute values for pjimg.sources.Noise subclasses."""
    return {
        'device': 'cpu',
        'seed': 'spam',
    }

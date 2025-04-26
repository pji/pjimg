"""
fixtures
~~~~~~~~

Common fixtures for pjimg.test_sources.
"""
import pytest as pt


@pt.fixture
def noise_attr_defaults(source_attr_defaults):
    """Default attribute values for pjimg.sources.Noise subclasses."""
    return {
        'seed': None,
        **source_attr_defaults
    }


@pt.fixture
def noise_attr_set(source_attr_set):
    """"Changed attribute values for pjimg.sources.Noise subclasses."""
    return {
        'seed': 'spam',
        **source_attr_set
    }


@pt.fixture
def source_attr_defaults():
    """Default attribute values for pjimg.sources.Source subclasses."""
    return {
        'device': '',
        'force_device': False,
    }


@pt.fixture
def source_attr_set():
    """Changed attribute values for pjimg.sources.Source subclasses."""
    return {
        'device': 'cpu',
        'force_device': True,
    }

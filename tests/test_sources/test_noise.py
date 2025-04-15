"""
test_noise
~~~~~~~~~~

Unit tests for the :mod:`pjimg.sources.noise` module.
"""
import numpy as np

from pjimg.sources import noise as n


class TestNoise:
    # Test for initialization.
    def test_init_required(self):
        """When given no parameters, :class:`Noise` should initialize a new
        object with default attribute values.
        """
        optionals = {'seed': None, 'device': '',}
        obj = n.Noise()
        for attr in optionals:
            assert getattr(obj, attr) == optionals[attr]

    def test_init_optional(self):
        """When given parameters, :class:`Noise` should initialize a new
        object setting attributes to the given values.
        """
        optionals = {'seed': 'spam', 'device': 'cpu',}
        obj = n.Noise(**optionals)
        for attr in optionals:
            assert getattr(obj, attr) == optionals[attr]

    # Tests for fill.
    def test_fill(self):
        """When given the size of an array, :meth:`Noise.fill` should return
        an array that contains randomly generated noise.
        """
        noise = n.Noise(seed='spam')
        result = noise.fill((2, 8, 8))
        assert ((result * 0xff).astype(np.uint8) == np.array([
            [
                [0xb6, 0xe0, 0xc4, 0x94, 0x3e, 0x0a, 0xc6, 0x8c],
                [0xb2, 0x14, 0x1f, 0x1f, 0x3e, 0x2e, 0x08, 0x92],
                [0xaa, 0xb6, 0x9d, 0x57, 0xf4, 0xb7, 0xba, 0x1c],
                [0x52, 0x89, 0xe5, 0xdb, 0x7d, 0xc7, 0x52, 0x2b],
                [0x15, 0xc4, 0xb9, 0x46, 0xca, 0x44, 0x01, 0xae],
                [0x48, 0xee, 0x63, 0x8b, 0xf7, 0xbc, 0xa5, 0x0a],
                [0x8c, 0x21, 0xf7, 0x71, 0x99, 0x2c, 0xa9, 0x8a],
                [0x99, 0xa8, 0xba, 0xd9, 0x0b, 0xd8, 0x85, 0xc9],
            ],
            [
                [0xe4, 0x10, 0xc0, 0xf3, 0xf5, 0x17, 0xf4, 0x93],
                [0xd7, 0x72, 0x80, 0xd2, 0x6a, 0xc8, 0x5d, 0xee],
                [0xb7, 0xce, 0x10, 0x27, 0x7d, 0x7f, 0xe5, 0xfd],
                [0x5d, 0x91, 0xb4, 0x01, 0x78, 0x02, 0x5d, 0x1b],
                [0x04, 0x20, 0xb8, 0x23, 0x50, 0xc2, 0x67, 0x45],
                [0x94, 0x12, 0x72, 0x00, 0x67, 0x22, 0x63, 0xa4],
                [0x66, 0x79, 0x77, 0xa5, 0xf8, 0xcf, 0x46, 0xc2],
                [0xe6, 0x73, 0xa0, 0xa5, 0xb4, 0x16, 0x04, 0x4c],
            ],
        ], dtype=np.uint8)).all()

    def test_fill_different_seed_different_noise(self):
        """When given different seeds, two instances of :class:`Noise`
        should return different noise.
        """
        a = n.Noise('spam')
        b = n.Noise('eggs')
        shape = (2, 8, 8)
        assert not (a.fill(shape) == b.fill(shape)).all()

    def test_fill_same_seed_same_noise(self):
        """When given the same seeds, two instances of :class:`Noise`
        should return the same noise.
        """
        a = n.Noise('spam')
        b = n.Noise('spam')
        shape = (2, 8, 8)
        assert (a.fill(shape) == b.fill(shape)).all()

    def test_fill_no_seed_different_noise(self):
        """When given no seeds, two instances of :class:`Noise`
        should return different noise.

        .. note:
            This test is not deterministic. It's theoretically possible
            for this test to fail because the output of the unseeded
            random number generator is not predictable. This should be
            extremely unlikely.
        """
        a = n.Noise()
        b = n.Noise()
        shape = (2, 8, 8)
        assert not (a.fill(shape) == b.fill(shape)).all()


class TestNoiseTorch:
    # Tests for fill.
    def test_fill(self):
        """When given the size of an array, :meth:`Noise.fill` should return
        an array that contains randomly generated noise.
        """
        noise = n.Noise(seed='spam', device='cpu')
        result = noise.fill((2, 8, 8))
        assert ((result * 0xff).astype(np.uint8) == np.array([
            [
                [
                    [0x85, 0x96, 0x29, 0xcc, 0xc9, 0xf0, 0x74, 0x21],
                    [0xf4, 0x0d, 0xfe, 0x4d, 0x06, 0x67, 0x6d, 0xd2],
                    [0xde, 0x12, 0x02, 0x91, 0x27, 0x2a, 0x78, 0x53],
                    [0x2a, 0x88, 0x57, 0xbd, 0x3a, 0xea, 0x52, 0xde],
                    [0xf2, 0xb4, 0x84, 0x6f, 0x47, 0x29, 0xbc, 0xf2],
                    [0x1f, 0x03, 0x4b, 0x02, 0xa2, 0xc6, 0x62, 0x0e],
                    [0x6a, 0xd8, 0x6e, 0x39, 0xa1, 0xe0, 0x44, 0x02],
                    [0xbc, 0xf4, 0x5a, 0x03, 0xf9, 0xc4, 0x4c, 0xfe],
                ],
                [
                    [0x4c, 0x74, 0xfc, 0xc3, 0x70, 0xe7, 0x94, 0x0f],
                    [0x8d, 0x4e, 0x23, 0x2d, 0x75, 0x69, 0xd5, 0x15],
                    [0x10, 0xa7, 0x5f, 0xd5, 0xa7, 0x5c, 0x50, 0x03],
                    [0x3d, 0x5d, 0x5c, 0x3e, 0x52, 0xd5, 0xc9, 0x6b],
                    [0x07, 0xba, 0xce, 0xbc, 0xa3, 0x26, 0xc0, 0x67],
                    [0x22, 0x96, 0x62, 0xac, 0xa7, 0x4b, 0x14, 0x44],
                    [0xfa, 0xe7, 0x28, 0x32, 0x4f, 0x49, 0xd2, 0x65],
                    [0x9b, 0x25, 0x31, 0xfc, 0x1d, 0xcd, 0xd9, 0xe0],
                ],
            ],
        ], dtype=np.uint8)).all()

    def test_fill_different_seed_different_noise(self):
        """When given different seeds, two instances of :class:`Noise`
        should return different noise.
        """
        a = n.Noise('spam', device='cpu')
        b = n.Noise('eggs', device='cpu')
        shape = (2, 8, 8)
        assert not (a.fill(shape) == b.fill(shape)).all()

    def test_fill_same_seed_same_noise(self):
        """When given the same seeds, two instances of :class:`Noise`
        should return the same noise.
        """
        a = n.Noise('spam', device='cpu')
        b = n.Noise('spam', device='cpu')
        shape = (2, 8, 8)
        assert (a.fill(shape) == b.fill(shape)).all()

    def test_fill_no_seed_different_noise(self):
        """When given no seeds, two instances of :class:`Noise`
        should return different noise.

        .. note:
            This test is not deterministic. It's theoretically possible
            for this test to fail because the output of the unseeded
            random number generator is not predictable. This should be
            extremely unlikely.
        """
        a = n.Noise(device='cpu')
        b = n.Noise(device='cpu')
        shape = (2, 8, 8)
        assert not (a.fill(shape) == b.fill(shape)).all()


class TestEmbers:
    # Test for initialization.
    def test_init_required(self):
        """When given no parameters, :class:`Noise` should initialize a new
        object with default attribute values.
        """
        optionals = {
            'depth': 1,
            'threshold': 0.9998,
            'seed': None,
            'device': '',
        }
        obj = n.Embers()
        for attr in optionals:
            assert getattr(obj, attr) == optionals[attr]

    def test_init_optional(self):
        """When given parameters, :class:`Noise` should initialize a new
        object setting attributes to the given values.
        """
        optionals = {
            'depth': 4,
            'threshold': 0.5,
            'seed': 'spam',
            'device': 'cpu',
        }
        obj = n.Embers(**optionals)
        for attr in optionals:
            assert getattr(obj, attr) == optionals[attr]

    # Tests for fill.
    def test_fill(self):
        """When given the size of an array, :meth:`Noise.fill` should return
        an array that contains randomly generated noise.
        """
        noise = n.Embers(threshold=0.5, seed='bacon')
        result = noise.fill((2, 8, 8))
        assert ((result * 0xff).astype(np.uint8) == np.array([
            [
                [0xd7, 0x00, 0xdd, 0x00, 0x00, 0xc6, 0xd0, 0x00],
                [0xd1, 0x00, 0x00, 0x00, 0xc1, 0x00, 0xcc, 0xce],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc4, 0x00],
                [0x00, 0xce, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc4],
                [0xce, 0xd6, 0x00, 0x00, 0xc8, 0xc7, 0x00, 0x00],
                [0xde, 0x00, 0x00, 0xc7, 0x00, 0xcb, 0xc6, 0x00],
                [0x00, 0x00, 0xdb, 0x00, 0xca, 0xdf, 0xc4, 0x00],
                [0xcb, 0x00, 0x00, 0x00, 0x00, 0xc2, 0xd3, 0xcd],
            ],
            [
                [0xc0, 0xc7, 0xc5, 0x00, 0xca, 0xd7, 0x00, 0x00],
                [0xc8, 0xd1, 0xd1, 0x00, 0x00, 0x00, 0xc6, 0x00],
                [0xd5, 0x00, 0xdc, 0xca, 0xcd, 0x00, 0xc2, 0xc2],
                [0x00, 0xce, 0x00, 0xd9, 0xdd, 0x00, 0xd9, 0x00],
                [0x00, 0x00, 0x00, 0xd8, 0x00, 0xcd, 0x00, 0x00],
                [0x00, 0x00, 0xce, 0x00, 0xd9, 0xd8, 0xd3, 0xd8],
                [0xc8, 0xc5, 0xd3, 0xcd, 0xcb, 0xce, 0xc7, 0x00],
                [0xc1, 0xcc, 0xcb, 0x00, 0xd2, 0x00, 0xbf, 0x00],
            ],
        ], dtype=np.uint8)).all()


class TestEmbersTorch:
    # Tests for fill.
    def test_fill(self):
        """When given the size of an array, :meth:`Noise.fill` should return
        an array that contains randomly generated noise.
        """
        noise = n.Embers(threshold=0.5, seed='spam', device='cpu')
        result = noise.fill((2, 8, 8))
        assert ((result * 0xff).astype(np.uint8) == np.array([
            [
                [0xc0, 0xc5, 0x00, 0xd2, 0xd1, 0xdb, 0x00, 0x00],
                [0xdc, 0x00, 0xde, 0x00, 0x00, 0x00, 0x00, 0xd4],
                [0xd7, 0x00, 0x00, 0xc3, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0xc1, 0x00, 0xce, 0x00, 0xd9, 0x00, 0xd7],
                [0xdb, 0xcc, 0xc0, 0x00, 0x00, 0x00, 0xce, 0xdb],
                [0x00, 0x00, 0x00, 0x00, 0xc7, 0xd1, 0x00, 0x00],
                [0x00, 0xd5, 0x00, 0x00, 0xc7, 0xd7, 0x00, 0x00],
                [0xce, 0xdc, 0x00, 0x00, 0xdd, 0xd0, 0x00, 0xde],
            ],
            [
                [0x00, 0x00, 0xde, 0xd0, 0x00, 0xd9, 0xc4, 0x00],
                [0xc2, 0x00, 0x00, 0x00, 0x00, 0x00, 0xd4, 0x00],
                [0x00, 0xc9, 0x00, 0xd4, 0xc9, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0xd4, 0xd1, 0x00],
                [0x00, 0xce, 0xd2, 0xce, 0xc8, 0x00, 0xcf, 0x00],
                [0x00, 0xc4, 0x00, 0xca, 0xc9, 0x00, 0x00, 0x00],
                [0xdd, 0xd9, 0x00, 0x00, 0x00, 0x00, 0xd4, 0x00],
                [0xc6, 0x00, 0x00, 0xde, 0x00, 0xd2, 0xd5, 0xd7],
            ],
        ], dtype=np.uint8)).all()

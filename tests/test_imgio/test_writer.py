"""
test_writer
~~~~~~~~~~~

Unit tests for :mod:`pjimg.imgio.writer`.
"""
import pytest as pt

from pjimg import imgio as pjio


# Common test code for write.
def write_test(a, exp_name, ext, tmp_path):
    """Common test for :func:`pjimg.imgio.writer.write`."""
    path = tmp_path / f'spam.{ext}'
    pjio.write(path, a)

    with open(path, 'rb') as fh:
        saved = fh.read()
    with open(f'tests/test_imgio/data/{exp_name}.{ext}', 'rb') as fh:
        expected = fh.read()
    try:
        assert saved == expected
    except Exception as ex:
        cls = type(ex)
        raise cls(f'Type: {ext}. {str(ex)}')


# Test cases.
def test_save_8_bit_rgb_image(tmp_path):
    """Given image data in the 8-bit RGB color space and a file
    path ending with valid file type extension, :func:`save` should
    save the image data to the file path as the valid file type.
    """
    exts = [
        key for key in pjio.VALID_FORMATS
        if isinstance(pjio.VALID_FORMATS[key], pjio.Image)
    ]
    a = [
        [
            [
                [0xff, 0x7f, 0x00,],
                [0xff, 0x7f, 0x00,],
                [0xff, 0x7f, 0x00,],
            ],
            [
                [0x7f, 0x00, 0xff,],
                [0x7f, 0x00, 0xff,],
                [0x7f, 0x00, 0xff,],
            ],
            [
                [0x00, 0xff, 0x7f,],
                [0x00, 0xff, 0x7f,],
                [0x00, 0xff, 0x7f,],
            ],
        ],
    ]
    exp_name = '__test_save_rgb_image'
    for ext in exts:
        write_test(a, exp_name, ext, tmp_path)


def test_save_float_rgb_image(tmp_path):
    """Given image data in the floating point color space and a file
    path ending with valid file type extension, :func:`save` should
    save the image data to the file path as the valid file type.
    """
    exts = [
        key for key in pjio.VALID_FORMATS
        if isinstance(pjio.VALID_FORMATS[key], pjio.Image)
    ]
    a = [[
        [
            [1., .5, 0.,],
            [1., .5, 0.,],
            [1., .5, 0.,],
        ],
        [
            [.5, 0., 1.,],
            [.5, 0., 1.,],
            [.5, 0., 1.,],
        ],
        [
            [0., 1., .5,],
            [0., 1., .5,],
            [0., 1., .5,],
        ],
    ],]
    exp_name = '__test_save_rgb_image'
    for ext in exts:
        write_test(a, exp_name, ext, tmp_path)


def test_save_float_grayscale_image(tmp_path):
    """Given image data in the floating point grayscale space and a
    file path ending with valid file type extension, :func:`save`
    should save the image data to the file path as the valid file
    type.
    """
    exts = [
        key for key in pjio.VALID_FORMATS
        if isinstance(pjio.VALID_FORMATS[key], pjio.Image)
    ]
    a = [[
        [0., .5, 1.,],
        [0., .5, 1.,],
        [0., .5, 1.,],
    ],]
    exp_name = '__test_save_grayscale_image'
    for ext in exts:
        write_test(a, exp_name, ext, tmp_path)


def test_save_8_bit_grayscale_image(tmp_path):
    """Given image data in the 8 bit grayscale space and a
    file path ending with valid file type extension, :func:`save`
    should save the image data to the file path as the valid file
    type.
    """
    exts = [
        key for key in pjio.VALID_FORMATS
        if isinstance(pjio.VALID_FORMATS[key], pjio.Image)
    ]
    a = [[
        [0x00, 0x7f, 0xff],
        [0x00, 0x7f, 0xff],
        [0x00, 0x7f, 0xff],
    ],]
    exp_name = '__test_save_grayscale_image'
    for ext in exts:
        write_test(a, exp_name, ext, tmp_path)

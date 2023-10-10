"""
test_writer
~~~~~~~~~~~

Unit tests for :mod:`pjimg.imgio.writer`.
"""
import numpy as np
import pytest as pt

from pjimg import imgio as pjio


# Common tests.
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


def save_video_test(a, ext, codec, exp_name, tmp_path):
    """The common test code for :func:`imgwriter.save_video`."""
    path = tmp_path / f'spam.{ext}'
    pjio.write_video(path, a, 12, codec)

    with open(path, 'rb') as fh:
        saved = fh.read()
    with open(f'tests/data/{exp_name}', 'rb') as fh:
        expected = fh.read()
    try:
        assert saved == expected
    except Exception as ex:
        cls = type(ex)
        raise cls(f'Type: {ext} {codec}. {str(ex)}')


# Fixtures.
@pt.fixture
def float_grayscale_mutlifile(request, tmp_path):
    """Save float grayscale data as an image to multiple files."""
    marker = request.node.get_closest_marker('ext')
    ext = marker.args[0]
    a = [
        [
            [0., .5, 1.,],
            [0., .5, 1.,],
            [0., .5, 1.,],
        ],
        [
            [0., .5, 1.,],
            [0., .5, 1.,],
            [0., .5, 1.,],
        ],
    ]
    path = tmp_path / f'spam.{ext}'
    pjio.write(path, a)

    with open(tmp_path / f'spam_0.{ext}', 'rb') as fh:
        saved0 = fh.read()
    with open(tmp_path / f'spam_1.{ext}', 'rb') as fh:
        saved1 = fh.read()
    return saved0, saved1


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


def test_save_float_as_jpeg_not_series(tmp_path):
    """Given image data in the floating point RGB color space
    and a file path ending with "JPG", :func:`save` should save
    the image data to the file path as a JPEG file. If the image
    is not a series, the three dimensional array should still
    be saved as an image.
    """
    a = [
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
    ]
    path = tmp_path / 'spam_single.jpg'
    pjio.write(path, a, as_series=False)

    with open(path, 'rb') as fh:
        saved = fh.read()
    assert saved == (
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01'
        b'\x00\x00\xff\xdb\x00C\x00\x02\x01\x01\x01\x01\x01\x02\x01\x01'
        b'\x01\x02\x02\x02\x02\x02\x04\x03\x02\x02\x02\x02\x05\x04\x04'
        b'\x03\x04\x06\x05\x06\x06\x06\x05\x06\x06\x06\x07\t\x08\x06\x07'
        b'\t\x07\x06\x06\x08\x0b\x08\t\n\n\n\n\n\x06\x08\x0b\x0c\x0b\n'
        b'\x0c\t\n\n\n\xff\xdb\x00C\x01\x02\x02\x02\x02\x02\x02\x05\x03'
        b'\x03\x05\n\x07\x06\x07\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'
        b'\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\xff'
        b'\xc0\x00\x11\x08\x00\x03\x00\x03\x03\x01"\x00\x02\x11\x01\x03'
        b'\x11\x01\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01'
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08'
        b'\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05'
        b'\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06'
        b'\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82'
        b'\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijst'
        b'uvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97'
        b'\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5'
        b'\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3'
        b'\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9'
        b'\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xc4\x00\x1f\x01'
        b'\x00\x03\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00'
        b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x11'
        b'\x00\x02\x01\x02\x04\x04\x03\x04\x07\x05\x04\x04\x00\x01\x02w\x00'
        b'\x01\x02\x03\x11\x04\x05!1\x06\x12AQ\x07aq\x13"2\x81\x08\x14B\x91'
        b'\xa1\xb1\xc1\t#3R\xf0\x15br\xd1\n\x16$4\xe1%\xf1\x17\x18\x19\x1a'
        b'&\'()*56789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x82\x83\x84\x85\x86'
        b'\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4'
        b'\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2'
        b'\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9'
        b'\xda\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf2\xf3\xf4\xf5\xf6\xf7'
        b'\xf8\xf9\xfa\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?'
        b'\x00\xf5\x9f\xf8g\xbf\x87?\xf3\xf9\xe2\xaf\xfc/\xb5\x8f\xfeJ\xa2'
        b'\x8a+\xf3\xbf\xedl\xd7\xfe\x7f\xcf\xff\x00\x02\x97\xf9\x9f\xe4'
        b'\x1f\xfcG\xaf\x1c\xff\x00\xe8\xa9\xcc\xbf\xf0\xbb\x15\xff\x00\xcb'
        b'O\xff\xd9'
    )


@pt.mark.ext('jpg')
def test_save_float_grayscale_as_multiple_jpg(float_grayscale_mutlifile):
    """Given three dimensional image data in the floating point
    grayscale color space and a file path, :func:`save` should
    save the image data as multiple images to the file path as
    JPEG files.
    """
    saved0, saved1 = float_grayscale_mutlifile
    assert saved0 == (
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01'
        b'\x00\x00\xff\xdb\x00C\x00\x02\x01\x01\x01\x01\x01\x02\x01\x01'
        b'\x01\x02\x02\x02\x02\x02\x04\x03\x02\x02\x02\x02\x05\x04\x04'
        b'\x03\x04\x06\x05\x06\x06\x06\x05\x06\x06\x06\x07\t\x08\x06\x07'
        b'\t\x07\x06\x06\x08\x0b\x08\t\n\n\n\n\n\x06\x08\x0b\x0c\x0b\n'
        b'\x0c\t\n\n\n\xff\xc0\x00\x0b\x08\x00\x03\x00\x03\x01\x01\x11'
        b'\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01'
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07'
        b'\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04'
        b'\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05'
        b'\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R'
        b'\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHI'
        b'JSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92'
        b'\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8'
        b'\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5'
        b'\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1'
        b'\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6'
        b'\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfd=\xff'
        b'\x00\x82\x03\xff\x00\xca\x1b?g\xbf\xfb\'\xf0\xff\x00\xe8\xe9k'
        b'\xff\xd9'
    )
    assert saved1 == (
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01'
        b'\x00\x00\xff\xdb\x00C\x00\x02\x01\x01\x01\x01\x01\x02\x01\x01'
        b'\x01\x02\x02\x02\x02\x02\x04\x03\x02\x02\x02\x02\x05\x04\x04'
        b'\x03\x04\x06\x05\x06\x06\x06\x05\x06\x06\x06\x07\t\x08\x06\x07'
        b'\t\x07\x06\x06\x08\x0b\x08\t\n\n\n\n\n\x06\x08\x0b\x0c\x0b\n'
        b'\x0c\t\n\n\n\xff\xc0\x00\x0b\x08\x00\x03\x00\x03\x01\x01\x11'
        b'\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01'
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07'
        b'\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04'
        b'\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05'
        b'\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R'
        b'\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHI'
        b'JSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92'
        b'\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8'
        b'\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5'
        b'\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1'
        b'\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6'
        b'\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfd=\xff'
        b'\x00\x82\x03\xff\x00\xca\x1b?g\xbf\xfb\'\xf0\xff\x00\xe8\xe9k'
        b'\xff\xd9'
    )


@pt.mark.ext('png')
def test_save_float_grayscale_as_multiple_png(float_grayscale_mutlifile):
    """Given three dimensional image data in the floating point
    grayscale color space and a file path, :func:`save` should
    save the image data as multiple images to the file path as
    PNG files.
    """
    saved0, saved1 = float_grayscale_mutlifile
    assert saved0 == (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x03\x00\x00'
        b'\x00\x03\x08\x00\x00\x00\x00sC\xeac\x00\x00\x00\x14IDAT\x08'
        b'\x1dcd\xa8o`d\xa8o`d\xa8o\x00\x00\x10\x92\x03\x01\xe0\x97\x89'
        b'\x82\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    assert saved1 == (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x03\x00\x00'
        b'\x00\x03\x08\x00\x00\x00\x00sC\xeac\x00\x00\x00\x14IDAT\x08'
        b'\x1dcd\xa8o`d\xa8o`d\xa8o\x00\x00\x10\x92\x03\x01\xe0\x97\x89'
        b'\x82\x00\x00\x00\x00IEND\xaeB`\x82'
    )


@pt.mark.ext('tiff')
def test_save_float_grayscale_as_multiple_tiff(float_grayscale_mutlifile):
    """Given three dimensional image data in the floating point
    grayscale color space and a file path, :func:`save` should
    save the image data as multiple images to the file path as
    TIFF files.
    """
    saved0, saved1 = float_grayscale_mutlifile
    assert saved0 == (
        b'II*\x00\x12\x00\x00\x00\x80\x00\x0f\xe8\x08\x14\x12\x07'
        b'\x01\x00\x0c\x00\x00\x01\x03\x00\x01\x00\x00\x00\x03\x00'
        b'\x00\x00\x01\x01\x03\x00\x01\x00\x00\x00\x03\x00\x00\x00'
        b'\x02\x01\x03\x00\x01\x00\x00\x00\x08\x00\x00\x00\x03\x01'
        b'\x03\x00\x01\x00\x00\x00\x05\x00\x00\x00\x06\x01\x03\x00'
        b'\x01\x00\x00\x00\x01\x00\x00\x00\x11\x01\x04\x00\x01\x00'
        b'\x00\x00\x08\x00\x00\x00\x15\x01\x03\x00\x01\x00\x00\x00'
        b'\x01\x00\x00\x00\x16\x01\x03\x00\x01\x00\x00\x00\x03\x00'
        b'\x00\x00\x17\x01\x04\x00\x01\x00\x00\x00\t\x00\x00\x00'
        b'\x1c\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00=\x01\x03'
        b'\x00\x01\x00\x00\x00\x02\x00\x00\x00S\x01\x03\x00\x01\x00'
        b'\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00'
    )
    assert saved1 == (
        b'II*\x00\x12\x00\x00\x00\x80\x00\x0f\xe8\x08\x14\x12\x07'
        b'\x01\x00\x0c\x00\x00\x01\x03\x00\x01\x00\x00\x00\x03\x00'
        b'\x00\x00\x01\x01\x03\x00\x01\x00\x00\x00\x03\x00\x00\x00'
        b'\x02\x01\x03\x00\x01\x00\x00\x00\x08\x00\x00\x00\x03\x01'
        b'\x03\x00\x01\x00\x00\x00\x05\x00\x00\x00\x06\x01\x03\x00'
        b'\x01\x00\x00\x00\x01\x00\x00\x00\x11\x01\x04\x00\x01\x00'
        b'\x00\x00\x08\x00\x00\x00\x15\x01\x03\x00\x01\x00\x00\x00'
        b'\x01\x00\x00\x00\x16\x01\x03\x00\x01\x00\x00\x00\x03\x00'
        b'\x00\x00\x17\x01\x04\x00\x01\x00\x00\x00\t\x00\x00\x00'
        b'\x1c\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00=\x01\x03'
        b'\x00\x01\x00\x00\x00\x02\x00\x00\x00S\x01\x03\x00\x01\x00'
        b'\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00'
    )


# Tests for save_video.
def test_save_video_float_grayscale(tmp_path):
    """Given video data in the floating point grayscale space, a
    file path ending with valid file type extension, and a codec
    valid for the file type, :func:`save_video` should save the
    video data to the file path as the valid file type.
    """
    a = np.zeros((3, 480, 720), dtype=float)
    a[1, :, :] = 0.5
    a[2, :, :] = 1.0
    vids = [vid for vid in pjio.VALID_FORMATS if isinstance(vid, pjio.Video)]
    for vid in vids:
        for codec in vid.codecs:
            exp_name = f'__test_save_grayscale_video_{codec}.{vid.ext}'
            save_video_test(a, vid.ext, codec, exp_name, tmp_path)


def test_save_video_8_bit_grayscale(tmp_path):
    """Given video data in the floating point grayscale space, a
    file path ending with valid file type extension, and a codec
    valid for the file type, :func:`save_video` should save the
    video data to the file path as the valid file type.
    """
    a = np.zeros((3, 480, 720), dtype=np.uint8)
    a[1, :, :] = 0x7f
    a[2, :, :] = 0xff
    vids = [vid for vid in pjio.VALID_FORMATS if isinstance(vid, pjio.Video)]
    for vid in vids:
        for codec in vid.codecs:
            exp_name = f'__test_save_grayscale_video_{codec}.{vid.ext}'
            save_video_test(a, vid.ext, codec, exp_name, tmp_path)


def test_save_video_float_rgb(tmp_path):
    """Given video data in the floating point RGB space, a
    file path ending with valid file type extension, and a codec
    valid for the file type, :func:`save_video` should save the
    video data to the file path as the valid file type.
    """
    a = np.zeros((3, 480, 720, 3), dtype=float)
    a[0, :, :, 0] = 1.0
    a[1, :, :, 1] = 1.0
    a[2, :, :, 2] = 1.0
    vids = [vid for vid in pjio.VALID_FORMATS if isinstance(vid, pjio.Video)]
    for vid in vids:
        for codec in vid.codecs:
            exp_name = f'__test_save_rgb_video_{codec}.{vid.ext}'
            save_video_test(a, vid.ext, codec, exp_name, tmp_path)


def test_save_video_8_bit_rgb(tmp_path):
    """Given video data in the 8-bit RGB space, a file path ending
    with valid file type extension, and a codec valid for the file
    type, :func:`save_video` should save the video data to the file
    path as the valid file type.
    """
    a = np.zeros((3, 480, 720, 3), dtype=np.uint8)
    a[0, :, :, 0] = 0xff
    a[1, :, :, 1] = 0xff
    a[2, :, :, 2] = 0xff
    vids = [vid for vid in pjio.VALID_FORMATS if isinstance(vid, pjio.Video)]
    for vid in vids:
        for codec in vid.codecs:
            exp_name = f'__test_save_rgb_video_{codec}.{vid.ext}'
            save_video_test(a, vid.ext, codec, exp_name, tmp_path)

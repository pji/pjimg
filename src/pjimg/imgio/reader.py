"""
Reading Data from Files
=======================

Read images and video data from files for manipulation.

.. autofunction:: pjimg.imgio.read
.. autofunction:: pjimg.imgio.read_tensor
.. autofunction:: pjimg.imgio.read_image
.. autofunction:: pjimg.imgio.read_image_tensor
.. autofunction:: pjimg.imgio.read_video
.. autofunction:: pjimg.imgio.read_video_tensor

"""
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from pjimg.imgio.constants import VALID_FORMATS
from pjimg.imgio.model import Image, UnsupportedFileType, Video
from pjimg.util import AnyAry, ImgAry, ImgTnsr, IntAry


# Core functions.
def read(path: Union[str, Path]) -> ImgAry:
    """Read an image or video file.

    .. note:
        :mod:`pjimg` works with both image and video data. To standardize
        the output between the two types of data, it treats still images
        as a single frame video. As a result, it will add a Z axis to image
        data from still images.

    :param path: The path to the file.
    :return: The image or video data as a :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    :usage:
        To read an image file:

            >>> filepath = 'tests/data/__test_save_rgb_image.tiff'
            >>> read_image(filepath)
            array([[[[1.        , 0.49803922, 0.        ],
                     [1.        , 0.49803922, 0.        ],
                     [1.        , 0.49803922, 0.        ]],
            <BLANKLINE>
                    [[0.49803922, 0.        , 1.        ],
                     [0.49803922, 0.        , 1.        ],
                     [0.49803922, 0.        , 1.        ]],
            <BLANKLINE>
                    [[0.        , 1.        , 0.49803922],
                     [0.        , 1.        , 0.49803922],
                     [0.        , 1.        , 0.49803922]]]])

    """
    path = Path(path)
    ftype = VALID_FORMATS[path.suffix.casefold()[1:]]
    if isinstance(ftype, Image):
        a = read_image(path)
    elif isinstance(ftype, Video):
        a = read_video(path)
    else:
        raise UnsupportedFileType(f'{path.suffix}')
    return a


def read_to_tensor(
    path: Union[str, Path],
    device: str = 'cpu'
) -> ImgTnsr:
    """Read an image or video file to a tensor.

    .. note:
        :mod:`pjimg` works with both image and video data. To standardize
        the output between the two types of data, it treats still images
        as a single frame video. As a result, it will add a Z axis to image
        data from still images.

    :param path: The path to the file.
    :param as_video: (Optional.) Whether the data should be read as
        a still image or a single frame of video. The difference is
        video has one more dimension than a still image.
    :param device: (Optional.) Determines the device for the read
        image data. Defaults to `cpu`, which is the systems' CPU.
    :return: The image or video data as a :class:`torch.Tensor`.
    :rtype: torch.Tensor
    :usage:
        To read an image file, getting back a tensor:

            >>> filepath = 'tests/test_imgio/data/__test_save_rgb_image.tiff'
            >>> read_image(filepath)
            tensor([[[[1.0000, 0.4980, 0.0000],
                      [1.0000, 0.4980, 0.0000],
                      [1.0000, 0.4980, 0.0000]],
            <BLANKLINE>
                     [[0.4980, 0.0000, 1.0000],
                      [0.4980, 0.0000, 1.0000],
                      [0.4980, 0.0000, 1.0000]],
            <BLANKLINE>
                     [[0.0000, 1.0000, 0.4980],
                      [0.0000, 1.0000, 0.4980],
                      [0.0000, 1.0000, 0.4980]]]])

    """
    path = Path(path)
    ftype = VALID_FORMATS[path.suffix.casefold()[1:]]
    if isinstance(ftype, Image):
        t = read_image_to_tensor(path, device=device)
    elif isinstance(ftype, Video):
        t = read_video_to_tensor(path, device=device)
    else:
        raise UnsupportedFileType(f'{path.suffix}')
    return t


def read_image(
    filepath: Union[str, Path],
    as_video: bool = True
) -> ImgAry:
    """Read image data from an image file.

    .. note:
        :mod:`pjimg` works with both image and video data. To standardize
        the output between the two types of data, it treats still images
        as a single frame video. As a result, it will add a Z axis to image
        data from still images.

    :param filepath: The location of the image file to read.
    :param as_video: (Optional.) Whether the data should be read as
        a still image or a single frame of video. The difference is
        video has one more dimension than a still image.
    :return: A :class:`numpy.ndarray` object.
    :rtype: numpy.ndarray
    :usage:
        To read an image file:

            >>> filepath = 'tests/data/__test_save_rgb_image.tiff'
            >>> read_image(filepath)
            array([[[[1.        , 0.49803922, 0.        ],
                     [1.        , 0.49803922, 0.        ],
                     [1.        , 0.49803922, 0.        ]],
            <BLANKLINE>
                    [[0.49803922, 0.        , 1.        ],
                     [0.49803922, 0.        , 1.        ],
                     [0.49803922, 0.        , 1.        ]],
            <BLANKLINE>
                    [[0.        , 1.        , 0.49803922],
                     [0.        , 1.        , 0.49803922],
                     [0.        , 1.        , 0.49803922]]]])

    """
    # Read in the raw image data.
    raw = _read_raw_image(filepath)

    # If the data in the file was unsigned 8-bit integers, convert it
    # to floats in the range 0 <= x <= 1.
    a = raw.astype(np.float64)
    if raw.dtype == np.uint8:
        a /= 0xff

    # Opencv returns color data from RGB files as BGR. Transform it
    # back to RGB.
    if len(a.shape) == 3:
        a = np.flip(a, -1)

    # Since this module deals with video and still images, it allows
    # you to read the image in as a single frame of video rather than
    # a still image. The difference is video has an additional
    # dimension.
    if as_video:
        a = a[np.newaxis, ...]
    return a


def read_image_to_tensor(
    filepath: str | Path,
    as_video: bool = True,
    device: str = 'cpu'
) -> ImgTnsr:
    """Read image data from an image file to a tensor.

    :param filepath: The location of the image file to read.
    :param as_video: (Optional.) Whether the data should be read as
        a still image or a single frame of video. The difference is
        video has one more dimension than a still image.
    :param device: (Optional.) Determines the device for the read
        image data. Defaults to `cpu`, which is the systems' CPU.
    :return: A :class:`torch.Tensor` object.
    :rtype: torch.Tensor
    :usage:
        To read an image file, getting back a tensor:

            >>> filepath = 'tests/test_imgio/data/__test_save_rgb_image.tiff'
            >>> read_image_to_tensor(filepath)
            tensor([[[[1.0000, 0.4980, 0.0000],
                      [1.0000, 0.4980, 0.0000],
                      [1.0000, 0.4980, 0.0000]],
            <BLANKLINE>
                     [[0.4980, 0.0000, 1.0000],
                      [0.4980, 0.0000, 1.0000],
                      [0.4980, 0.0000, 1.0000]],
            <BLANKLINE>
                     [[0.0000, 1.0000, 0.4980],
                      [0.0000, 1.0000, 0.4980],
                      [0.0000, 1.0000, 0.4980]]]])

    .. note:
        The imgwriter package works with both image and video data.
        In an attempt to standardize the output between the two types of
        data, it treats still images as a single frame video. As a result,
        it will add a Z axis to image data from still images.

    """
    # Read in the raw image data.
    a = _read_raw_image(filepath)
    t = torch.tensor(a, dtype=torch.float32, device=device)

    # If the data in the file was unsigned 8-bit integers, convert it
    # to floats in the range 0 <= x <= 1.
    if a.dtype == np.uint8:
        t /= 0xff

    # Opencv returns color data from RGB files as BGR. Transform it
    # back to RGB.
    if len(t.shape) == 3:
        t = torch.flip(t, dims=(-1,))

    # Since this module deals with video and still images, it allows
    # you to read the image in as a single frame of video rather than
    # a still image. The difference is video has an additional
    # dimension.
    if as_video:
        t = t.unsqueeze(0)
    return t


def read_video(path: Union[str, Path]) -> ImgAry:
    """Capture image data from a video file.

    .. note:
        Video saved with :func:`imgwriter.save` or most other methods
        will use a codec to compress the video. That compression is
        lossy. That means the array to read from the file will not be
        exactly the same as the array you saved out to the file.

    :param path: The path to the file to read.
    :return: A :class:`numpy.ndarray` containing the data from the file.
    :rtype: numpy.ndarray
    """
    capture = cv2.VideoCapture(str(path))
    frames = []
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)
    capture.release()
    a = np.zeros((len(frames), *frames[0].shape), dtype=frames[0].dtype)
    for i, frame in enumerate(frames):
        a[i] = frame
    return a


def read_video_to_tensor(path: str | Path, device: str = 'cpu') -> ImgTnsr:
    """Capture image data from a video file to a tensor.

    .. note:
        Video saved with :func:`imgwriter.save` or most other methods
        will use a codec to compress the video. That compression is
        lossy. That means the array to read from the file will not be
        exactly the same as the array you saved out to the file.

    :param path: The path to the file to read.
    :param device: (Optional.) Determines the device for the read
        image data. Defaults to `cpu`, which is the systems' CPU.
    :return: A :class:`torch.Tensor` object.
    :rtype: torch.Tensor
    """
    a = read_video(path)
    return torch.tensor(a, device=device)


# Private functions.
def _read_raw_image(filepath: str | Path) -> AnyAry:
    """Load the raw int array from an image file."""
    # Ensure filepath is a string in case opencv doesn't like Path.
    filepath = str(filepath)

    # Before wasting time trying to open the file, check if it
    # even exists.
    if not Path(filepath).is_file():
        msg = f'There is no file at {filepath}.'
        raise FileNotFoundError(msg)

    # Read in the data from the image file. Don't change whether it's
    # color or grayscale. If it wasn't readable, puke.
    raw = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if raw is None:
        msg = f'The file at {filepath} cannot be read.'
        raise ValueError(msg)
    return raw

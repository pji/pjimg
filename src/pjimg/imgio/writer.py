"""
Writing Data to Files
=====================

Write image and video data to files.

.. autofunction:: pjimg.imgio.write
.. autofunction:: pjimg.imgio.write_image
.. autofunction:: pjimg.imgio.write_video

"""
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from pjimg.imgio.constants import VALID_FORMATS
from pjimg.imgio.model import (
    Image, Video, UnsupportedFileType, Saver, WrappedSaver
)
from pjimg.util import ArrayLike, IntAry, X, Y, Z, float_to_uint8


# Decorators
def uses_opencv(fn: Saver) -> WrappedSaver:
    """Condition the image data for use by opencv prior to saving."""
    @wraps(fn)
    def wrapper(
        filepath: Union[str, Path], a: ArrayLike, *args, **kwargs
    ) -> None:
        # Convert the image data to an array just in case we were passed
        # something else.
        a = np.array(deepcopy(a))

        # While TIFFs can handle 32-bit floats, JPGs and PNGs can't, so
        # rather than having TIFFs as an exception, just convert all floats
        # to unsigned 8-bit integers.
        if a.dtype in [float, np.float32]:
            a = float_to_uint8(a)

        # If the data isn't a float but not a unsigned 8-bit integer,
        # we assume it's in the right scale. So, just convert to a
        # unsigned 8-bit integer.
        elif a.dtype != np.uint8:
            a = a.astype(np.uint8)

        # opencv saves color data in BGR order, so RGB data needs to be
        # flipped to BGR.
        if len(a.shape) == 4:
            a = np.flip(a, -1)
        elif (len(a.shape) == 3
                and 'as_series' in kwargs
                and not kwargs['as_series']):
            a = np.flip(a, -1)

        return fn(filepath, a, *args, **kwargs)
    return wrapper


# Image output functions.
def write(filepath: Union[str, Path], a: ArrayLike, *args, **kwargs) -> None:
    """Save an array of image data to file.

    :param filepath: The location and name of the file that will
        be saved. The file extension will determine the format used
        by the file.
    :param a: The array of image data.
    :return: None.
    :rtype: None.

    Supported File Types
    --------------------
    :mod:`pjimg.imgio` uses :mod:`cv2` for all save operations. The
    formats supported by :mod:`cv2` varies depending on operating
    system and software installed. Since I need to check the file
    type to determine whether an image or a video is being saved,
    I have to limit the supported file formats to ones I think
    are supported across platforms.
    """
    filepath = Path(filepath)
    ftype = filepath.suffix.casefold()[1:]
    save_as = VALID_FORMATS[ftype]
    if isinstance(save_as, Image):
        save_fn = write_image
    elif isinstance(save_as, Video):
        save_fn = write_video
    else:
        raise UnsupportedFileType(f'{ftype}')
    save_fn(filepath, a, *args, **kwargs)


@uses_opencv
def write_image(
    filepath: Union[str, Path],
    a: IntAry,
    as_series: bool = True
) -> None:
    """Save an array of image data as an image file.

    :param filepath: The location and name of the file that will
        be saved. The file extension will determine the format used
        by the file. The data needs to be either in an RGB or grayscale
        color space.
    :param a: The array of image data.
    :param as_series: (Optional.) Whether the array is intended to be a
        series of images.
    :return: None.
    :rtype: None.
    """
    filepath = Path(filepath)

    # If the array isn't a series of images, just save what is given.
    if not as_series:
        cv2.imwrite(str(filepath), a)

    # If there is just 1 item in the Z axis, save the image data as
    # a single image.
    elif a.shape[Z] == 1:
        a = a[Z]
        cv2.imwrite(str(filepath), a)

    # If there are multiple items in the Z axis, save the image data
    # as multiple images.
    else:
        fileparent = filepath.parent
        filename = filepath.stem
        filetype = filepath.suffix
        for i in range(a.shape[Z]):
            framepath = str(fileparent / f'{filename}_{i}{filetype}')
            cv2.imwrite(framepath, a[i])


@uses_opencv
def write_video(
    filepath: Union[str, Path],
    a: IntAry,
    framerate: float = 12.0,
    codec: str = 'mp4v'
) -> None:
    """Save an array of image data as a video file.

    :param filepath: The location and name of the file that will
        be saved. The file extension will determine the container
        type used for the file.
    :param a: The array of image data.
    :param framerate: (Optional.) The number of frames the video will
        play per second.
    :param codec: (Optional.) The codec used to encode the image data
        into video. The exact list of supported codecs depends upon
        the operating system. Per the opencv documentation, Linux and
        Windows will tend to use the list supported by ffmpeg and
        macOS will use the list suported by QTKit.
    :return: None.
    :rtype: None.
    """
    # cv2.VideoWriter requires a string rather than a Path.
    filepath = str(filepath)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    framesize = (a.shape[X], a.shape[Y])
    iscolor = False
    if len(a.shape) == 4:
        iscolor = True

    vwriter = cv2.VideoWriter(
        filepath, fourcc, framerate, framesize, iscolor
    )
    for i in range(a.shape[Z]):
        vwriter.write(a[i])
    vwriter.release()


"""
imgio
~~~~~

File I/O for saving images and video.

.. automodule:: pjimg.imgio.reader
.. automodule:: pjimg.imgio.writer
"""
from pjimg.imgio.constants import VALID_FORMATS
from pjimg.imgio.model import Image, Video
from pjimg.imgio.reader import (
    read,
    read_image,
    read_image_to_tensor,
    read_to_tensor,
    read_video,
    read_video_to_tensor
)
from pjimg.imgio.writer import write, write_image, write_video

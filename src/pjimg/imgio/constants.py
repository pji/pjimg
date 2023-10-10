"""
constants
~~~~~~~~~

Common constants used in :mod:`pjimg.imgio`.
"""
from typing import Union

from pjimg.imgio.model import Image, Video


# File formats supported by imgio.
VALID_FORMATS: dict[str, Union[Image, Video]] = {
    'bmp': Image('bmp', 'Windows bitmap'),
    'dib': Image('dib', 'Windows bitmap'),

    'hdr': Image('hdr', 'Radiance HDR'),
    'pic': Image('pic', 'Radiance HDR'),

    'jpe': Image('jpe', 'JPEG'),
    'jpg': Image('jpg', 'JPEG'),
    'jpeg': Image('jpeg', 'JPEG'),

    'png': Image('png', 'portable network graphics'),

    'pnm': Image('pnm', 'portable image format'),

    'ras': Image('ras', 'Sun raster'),
    'sr': Image('sr', 'Sun raster'),

    'tif': Image('tif', 'TIFF'),
    'tiff': Image('tiff', 'TIFF'),

    'webp': Image('webp', 'WebP'),

    'avi': Video('avi', 'Audio Video Interleave', ('avc1', 'mp4v',)),
    'mov': Video('mov', 'QuickTime movie', ('avc1', 'hev1', 'mp4v',)),
    'mp4': Video('mp4', 'MPEG-4 part 14', ('avc1', 'hev1', 'mp4v',)),
}

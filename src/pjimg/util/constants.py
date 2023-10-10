"""
constants
~~~~~~~~~

Common constants used in :mod:`pjimg`.
"""
# Exportable names.
__all__ = ['RESOLUTIONS', 'X', 'Y', 'Z',]

# Dimensional axes.
X, Y, Z = 2, 1, 0

# Common video pixel dimensions.
RESOLUTIONS: dict[str, tuple[int, int]] = {
    'dv_ntsc': (720, 480),
    'd1_ntsc': (720, 486),
    'dv_pal': (720, 576),
    'd1_pal': (720, 576),
    'dvcpro_hd_720p': (960, 720),
    'dvcpro_hd_1080_59i': (1280, 1080),
    'dvcpro_hd_1080_50i': (1440, 1080),
    'hdv_1080i': (1440, 1080),
    'hdv_1080p': (1440, 1080),
    'sony_hdcam': (1440, 1080),
    'sony_hdcam_sr': (1440, 1080),
    'academy_2x': (1828, 1332),
    'full_aperature_native_2x': (2048, 1556),
    'academy_4x': (3656, 2664),
    'full_aperature_4x': (4096, 3112),
    '720p': (1280, 720),
    '1080p': (1920, 1080),
    '4k': (3840, 2160),
    '8k': (7680, 4320),
    '16k': (15360, 8640),
}

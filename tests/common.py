"""
common
~~~~~~

General purpose tools used in tests.
"""
import numpy as np


# Utility functions.
def mkhex(a):
    return (a * 0xff).astype(np.uint8)

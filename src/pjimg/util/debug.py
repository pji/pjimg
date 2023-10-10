"""
debug
~~~~~

Utilities useful for debugging code that uses :mod:`pjimg`.
"""
import numpy as np

from pjimg.util.model import NumAry


# Functions.
def print_array(a: NumAry, depth: int = 0) -> None:
    """Write the values of the given array to stdout.
    
    :param a: The array to print.
    :param depth: (Optional.) How far to indent the printed lines.
    :return: None.
    :rtype: NoneType
    """
    if len(a.shape) > 1:
        print(' ' * (4 * depth) + '[')
        for i in range(a.shape[0]):
            print_array(a[i], depth + 1)
        print(' ' * (4 * depth) + '],')

    else:
        if a.dtype in [int, np.uint, np.uint8]:
            tmp = '0x{:02x}'
        elif a.dtype in [float, np.float32, np.float64]:
            tmp = '{:>1.4f}'
        else:
            tmp = '{}'
        nums = [tmp.format(n) for n in a]
        print(' ' * (4 * depth) + '[' + ', '.join(nums) + '],')

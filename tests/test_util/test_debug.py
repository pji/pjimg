"""
test_debug
~~~~~~~~~~

Unit tests for :mod:`pjimg.util.debug`.
"""
import numpy as np

import pjimg.util.debug as dbug


# Test cases.
class TestPrintArray:
    def test_print_array(self, capsys):
        """Given an array of numbers, :func:`print_array` should print
        the numbers in a structure that can be pasted into a unit test
        case.
        """
        a = np.array([[
            [0.0000, 0.5000, 1.000],
            [0.0000, 0.5000, 1.000],
        ]], dtype=float)
        dbug.print_array(a)
        captured = capsys.readouterr()
        assert captured.out == '\n'.join([
            '[',
            '    [',
            '        [0.0000, 0.5000, 1.0000],',
            '        [0.0000, 0.5000, 1.0000],',
            '    ],',
            '],',
            '',
        ])

    def test_print_array_uint(self, capsys):
        """Given an array of numbers, :func:`print_array` should print
        the numbers in a structure that can be pasted into a unit test
        case. If the numbers are integers, the numbers should be printed
        in hexadecimal.
        """
        a = np.array([[
            [0x00, 0x7f, 0xff],
            [0x00, 0x7f, 0xff],
        ]], dtype=np.uint8)
        dbug.print_array(a)
        captured = capsys.readouterr()
        assert captured.out == '\n'.join([
            '[',
            '    [',
            '        [0x00, 0x7f, 0xff],',
            '        [0x00, 0x7f, 0xff],',
            '    ],',
            '],',
            '',
        ])

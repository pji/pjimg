"""
constants
=========

Constant data values for :mod:`pjimg.filters`.
"""
from pjimg.filters.model import Color, ColorDict


# Color definition shortcuts for the colorize filter.
COLORS = ColorDict({
    # Grayscale
    'a': Color(('hsv(0, 0%, 100%)', 'hsv(0, 0%, 0%)')),
    'A': Color(('hsl(0, 0%, 75%)', 'hsl(0, 0%, 25%)')),

    # Electric blue.
    'b': Color(('hsv(200, 100%, 100%)', 'hsv(200, 100%, 0%)')),
    'B': Color(('hsl(200, 100%, 75%)', 'hsl(200, 100%, 25%)')),

    'b+': Color(('hsv(205, 100%, 100%)', 'hsv(200, 100%, 0%)')),
    'B+': Color(('hsl(205, 100%, 75%)', 'hsl(200, 100%, 25%)')),

    'bk': Color(('hsv(200, 30%, 20%)', 'hsv(200, 30%, 0%)')),
    'BK': Color(('hsl(200, 30%, 30%)', 'hsl(200, 30%, 10%)')),
    
    'bp': Color(('hsv(200, 100%, 100%)', 'hsv(320, 100%, 100%)')),
    'bc': Color(('hsv(200, 100%, 100%)', 'hsv(35, 100%, 100%)')),

    # Cream
    'c': Color(('hsv(35, 100%, 100%)', 'hsv(35, 100%, 0%)')),
    'C': Color(('hsl(35, 100%, 80%)', 'hsl(35, 100%, 25%)')),

    'c-': Color(('hsv(30, 100%, 100%)', 'hsv(35, 100%, 0%)')),
    'C-': Color(('hsl(30, 100%, 80%)', 'hsl(35, 100%, 25%)')),

    'c+': Color(('hsv(40, 100%, 100%)', 'hsv(35, 100%, 0%)')),
    'C+': Color(('hsl(40, 100%, 80%)', 'hsl(35, 100%, 25%)')),

    'ck': Color(('hsv(35, 30%, 20%)', 'hsv(35, 30%, 0%)')),
    'CK': Color(('hsl(35, 30%, 30%)', 'hsl(35, 30%, 10%)')),

    'cb': Color(('hsv(35, 100%, 100%)', 'hsv(200, 100%, 100%)')),
    'cp': Color(('hsv(35, 100%, 100%)', 'hsv(320, 100%, 100%)')),

    # Dark.
    'k': Color(('hsv(220, 30%, 20%)', 'hsv(220, 30%, 0%)')),
    'K': Color(('hsl(220, 30%, 30%)', 'hsl(220, 30%, 10%)')),

    'kk': Color(('hsv(220, 30%, 10%)', 'hsv(220, 30%, 0%)')),
    'KK': Color(('hsl(220, 30%, 15%)', 'hsl(220, 30%, 5%)')),

    'kp': Color(('hsv(220, 30%, 20%)', 'hsv(320, 30%, 20%)')),
    'kP': Color(('hsv(220, 30%, 20%)', 'hsv(320, 30%, 100%)')),

    # Ectoplasmic teal.
    'e': Color(("hsv(190, 50%, 100%)", "hsv(190, 100%, 0%)")),
    'E': Color(("hsl(190, 50%, 100%)", "hsl(190, 100%, 30%)")),

    # Electric green.
    'g': Color(('hsv(90, 100%, 100%)', 'hsv(90, 100%, 0%)')),
    'G': Color(('hsl(90, 100%, 75%)', 'hsl(90, 100%, 25%)')),

    'gk': Color(('hsv(90, 30%, 20%)', 'hsv(90, 30%, 0%)')),
    'GK': Color(('hsl(90, 30%, 30%)', 'hsl(90, 30%, 10%)')),

    # Slate.
    'l': Color(('hsv(220, 30%, 50%)', 'hsv(220, 30%, 0%)')),
    'L': Color(('hsl(220, 30%, 75%)', 'hsl(220, 30%, 25%)')),

    # Electric pink.
    'p': Color(('hsv(320, 100%, 100%)', 'hsv(320, 100%, 0%)')),
    'P': Color(('hsl(320, 100%, 75%)', 'hsl(320, 100%, 25%)')),

    'pb': Color(('hsv(320, 100%, 100%)', 'hsv(200, 100%, 100%)')),
    'pc': Color(('hsv(320, 100%, 100%)', 'hsv(35, 100%, 100%)')),

    # Royal purple.
    'r': Color(('hsv(280, 100%, 100%)', 'hsv(280, 100%, 0%)')),
    'R': Color(('hsl(280, 100%, 75%)', 'hsl(280, 100%, 25%)')),

    'rw': Color(('hsv(285, 100%, 100%)', 'hsv(280, 100%, 0%)')),
    'Rw': Color(('hsl(285, 100%, 75%)', 'hsl(280, 100%, 25%)')),

    # Scarlet.
    's': Color(('hsv(350, 100%, 100%)', 'hsv(10, 100%, 0%)')),
    'S': Color(('hsl(350, 100%, 75%)', 'hsl(10, 100%, 25%)')),

    'sw': Color(('hsv(0, 100%, 100%)', 'hsv(10, 100%, 0%)')),
    'Sw': Color(('hsl(0, 100%, 75%)', 'hsl(10, 100%, 25%)')),

    'sk': Color(('hsv(350, 30%, 20%)', 'hsv(10, 30%, 0%)')),
    'SK': Color(('hsl(350, 30%, 30%)', 'hsl(10, 30%, 10%)')),

    # White.
    'w': Color(('hsv(0, 0%, 100%)', 'hsv(0, 0%, 0%)')),
    'W': Color(('hsl(0, 0%, 75%)', 'hsl(0, 0%, 25%)')),

    # Hue templates.
    't': Color(('hsv({}, 100%, 100%)', 'hsv({}, 100%, 0%)')),
    'T': Color(('hsl({}, 100%, 75%)', 'hsl({}, 100%, 25%)')),
})

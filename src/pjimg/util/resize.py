"""
resize
~~~~~~

Functions for resizing numpy arrays through interpolation.

.. autofunction:: pjimg.util.build_resizing_matrices
.. autofunction:: pjimg.util.magnify_size
.. autofunction:: pjimg.util.pad_array
.. autofunction:: pjimg.util.resize_array
"""
from math import prod
from typing import Optional, Union

import numpy as np

from pjimg.util import lerps as lp
from pjimg.util.constants import X, X_, Y, Y_, Z, Z_
from pjimg.util.model import *


# Names available for import.
__all__ = [
    'bilinear_interpolation', 'build_resizing_matrices', 'magnify_size',
    'pad_array', 'resize_array', 'trilinear_interpolation',
]


# Public functions.
def bilinear_interpolation(a: ImgAry, factor: float) -> ImgAry:
    """Resize an two dimensional array using trilinear
    interpolation.

    :param a: The array to resize. The array is expected to have at
        least two dimensions.
    :param factor: The amount to resize the array. Given how the
        interpolation works, you probably don't get great results
        with factor less than or equal to .5. Consider multiple
        passes of interpolation with larger factors in those cases.
    :return: A :class:ndarray object.
    :rtype: numpy.ndarray
    """
    # Return the array unchanged if the array won't be magnified.
    if factor == 1:
        return a

    # Perform a defensive copy of the original array to avoid
    # unexpected side effects.
    a = a.copy()

    # Since we are magnifying the given array, the new array's shape
    # will increase by the magnification factor.
    mag_size = tuple(int(s * factor) for s in a.shape)

    # Map out the relationship between the old space and the
    # new space.
    indices = np.indices(mag_size)
    if factor > 1:
        whole = (indices // factor).astype(int)
        parts = (indices / factor - whole).astype(float)
    else:
        new_ends = [s - 1 for s in mag_size]
        old_ends = [s - 1 for s in a.shape]
        true_factors = [n / o for n, o in zip(new_ends, old_ends)]
        for i in range(len(true_factors)):
            if true_factors[i] == 0:
                true_factors[i] = .5
        whole = indices.copy()
        parts = indices.copy()
        for i in Y_, X_:
            whole[i] = (indices[i] // true_factors[i]).astype(int)
            parts[i] = (indices[i] / true_factors[i] - whole[i]).astype(float)
    del indices

    # Trilinear interpolation determines the value of a new pixel by
    # comparing the values of the eight old pixels that surround it.
    # The hashes are the keys to the dictionary that contains those
    # old pixel values. The key indicates the position of the pixel
    # on each axis, with one meaning the position is ahead of the
    # new pixel, and zero meaning the position is behind it.
    hashes = [f'{n:>02b}'[::-1] for n in range(2 ** 2)]
    hash_table = {}

    # The original array needs to be made one dimensional for the
    # numpy.take operation that will occur as we build the tables.
    raveled = np.ravel(a)

    # Build the table that contains the old pixel values to
    # interpolate.
    for hash in hashes:
        hash_whole = whole.copy()

        # Use the hash key to adjust the which old pixel we are
        # looking at.
        for axis in Y_, X_:
            if hash[axis] == '1':
                hash_whole[axis] += 1

                # Handle the pixels that were pushed off the far
                # edge of the original array by giving them the
                # value of the last pixel along that axis in the
                # original array.
                m = np.zeros(hash_whole[axis].shape, dtype=bool)
                m[hash_whole[axis] >= a.shape[axis]] = True
                hash_whole[axis][m] = a.shape[axis] - 1

        # Since numpy.take() only works in one dimension, we need to
        # map the three dimensional indices of the original array to
        # the one dimensional indices used by the raveled version of
        # that array.
        raveled_indices = hash_whole[Y_] * a.shape[X_]
        raveled_indices += hash_whole[X_]

        # Get the value of the pixel in the original array.
        hash_table[hash] = np.take(raveled, raveled_indices.astype(int))

    # Once the hash table has been built, clean up the working arrays
    # in case we are running short on memory.
    else:
        del hash_whole, raveled_indices, whole

    # Everything before this was to set up the interpolation. Now that
    # it's set up, we perform the interpolation. Since we are doing
    # this across three dimensions, it's a three stage process. Stage
    # one is along the X_ axis.
    x1 = lp.lerp(hash_table['00'], hash_table['01'], parts[X_])
    x2 = lp.lerp(hash_table['10'], hash_table['11'], parts[X_])

    # And stage three is along the Z axis. Since this is the last step
    # we can just return the result.
    return lp.lerp(x1, x2, parts[Y_])


def build_resizing_matrices(
    src_shape: Size,
    dst_shape: Size
) -> tuple[IntAry, IntAry, RatioAry]:
    """Create the indexing and distance arrays needed to interpolate
    values when resizing an array.

    :param src_shape: The original shape of the array.
    :param dst_shape: The resized shape of the array.
    :return: A :class:`tuple` object.
    :rtype: tuple
    """
    # Interpolation guesses a value between known data values. To
    # do this you need to know those points. The number of points
    # surrounding the value being guessed is the square of the
    # dimensions in the array.
    num_dim = len(src_shape)
    axes = range(num_dim)
    points = range(2 ** num_dim)

    # The relative positions of the points compared to the interpolated
    # value is coded by a binary text string where 1 is after the value
    # on the axis and 0 is before the value.
    rel_positions = _build_relative_position_masks(num_dim)

    # Create the map for position 0, which is before the interpolated
    # value on every axis.
    factors = _get_resizing_factors(src_shape, dst_shape)
    src_indices, x = _map_indices_and_distances(dst_shape, factors)

    # Create the maps for the rest of the positions.
    matrix_shape = (len(points) // 2, num_dim, *dst_shape)
    a = np.zeros(matrix_shape, dtype=int)
    b = a.copy()
    for pos in rel_positions:
        matrix_index = int(pos, 2) // 2
        pos_indices = src_indices.copy()
        for axis in axes:
            if pos[axis] == '1':
                pos_indices[axis] += 1

                # Cap the values in the array to the highest index in
                # the original array.
                cap = src_shape[axis] - 1
                pos_indices[pos_indices > cap] = cap

        # Put the value in the correct side of the resizing matrices.
        if pos.endswith('0'):
            a[matrix_index] = pos_indices
        else:
            b[matrix_index] = pos_indices

    # Return the arrays for the resizing interpolation.
    return a, b, x


def magnify_size(shape: Size, factor: int) -> Size:
    """Magnify the shape of an array.

    :param shape: The original shape of the array.
    :param factor: The magnification factor.
    :return: A :class:`tuple` containing the shape of the magnified array.
    """
    return tuple(int(n * factor) for n in shape)


def pad_array(
        a: NumAry,
        size: Size,
        fill: float = 0.0
    ) -> NumAry:
    """Pad an array to a larger size.
    
    :param a: The array to pad.
    :param size: The shape of the size.
    :param fill: The color of the padded area.
    :return: The padded :class:`numpy.ndarray`.
    :rtype: numpy.ndarray
    """
    # Create array at the new size.
    resized = np.full(size, fill, dtype=a.dtype)

    # Determine the amount the image has to be inset by in each dimension.
    size_diff = [n - o for n, o in zip(size, a.shape)]
    pad = [dim // 2 for dim in size_diff]
    end = [n + o for n, o in zip(pad, a.shape)]

    # Place the image and return.
    resized[pad[Z]:end[Z], pad[Y]:end[Y], pad[X]:end[X]] = a
    return resized


def resize_array(
    src: NumAry,
    shape: Size,
    interpolator: Interpolator = lp.ndlerp
) -> NumAry:
    """Resize a two dimensional array using an interpolation.

    :param src: The array to resize. The array is expected to have at
        least two dimensions.
    :param shape: The shape for the resized array.
    :param interpolator: The interpolation algorithm for the resizing.
    :return: A :class:`numpy.ndarray` object.
    :rtype: numpy.ndarray
    """
    # Perform defensive actions to prevent unneeded processing if
    # the array won't actually change and to make sure any changes
    # to the array won't have unexpected side effects.
    if shape == src.shape:
        return src
    src = src.copy()

    # Map out the relationship between the old space and the
    # new space.
    a_index, b_index, x = build_resizing_matrices(src.shape, shape)
    a = _replace_indices_with_values(src, a_index)
    b = _replace_indices_with_values(src, b_index)

    # Perform the interpolation using the mapped space and return.
    return interpolator(a, b, x)


def trilinear_interpolation(a: ImgAry, factor: float) -> ImgAry:
    """Resize an three dimensional array using trilinear
    interpolation.

    :param a: The array to resize. The array is expected to have at
        least three dimensions.
    :param factor: The amount to resize the array. Given how the
        interpolation works, you probably don't get great results
        with factor less than or equal to .5. Consider multiple
        passes of interpolation with larger factors in those cases.
    :return: A :class:ndarray object.
    :rtype: numpy.ndarray

    Usage::

        >>> import numpy as np
        >>>
        >>> a = np.array([
        ...     [
        ...             [0, 1],
        ...             [1, 0],
        ...     ],
        ...     [
        ...             [1, 0],
        ...             [0, 1],
        ...     ],
        ... ])
        >>> trilinear_interpolation(a, 2)
        array([[[0. , 0.5, 1. , 1. ],
                [0.5, 0.5, 0.5, 0.5],
                [1. , 0.5, 0. , 0. ],
                [1. , 0.5, 0. , 0. ]],
        <BLANKLINE>
               [[0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5]],
        <BLANKLINE>
               [[1. , 0.5, 0. , 0. ],
                [0.5, 0.5, 0.5, 0.5],
                [0. , 0.5, 1. , 1. ],
                [0. , 0.5, 1. , 1. ]],
        <BLANKLINE>
               [[1. , 0.5, 0. , 0. ],
                [0.5, 0.5, 0.5, 0.5],
                [0. , 0.5, 1. , 1. ],
                [0. , 0.5, 1. , 1. ]]])
    """
    # Return the array unchanged if the array won't be magnified.
    if factor == 1:
        return a

    # Perform a defensive copy of the original array to avoid
    # unexpected side effects.
    a = a.copy()

    # Since we are magnifying the given array, the new array's shape
    # will increase by the magnification factor.
    mag_size = tuple(int(s * factor) for s in a.shape)

    # Map out the relationship between the old space and the
    # new space.
    indices = np.indices(mag_size)
    if factor > 1:
        whole = (indices // factor).astype(int)
        parts = (indices / factor - whole).astype(float)
    else:
        new_ends = [s - 1 for s in mag_size]
        old_ends = [s - 1 for s in a.shape]
        true_factors = [n / o for n, o in zip(new_ends, old_ends)]
        for i in range(len(true_factors)):
            if true_factors[i] == 0:
                true_factors[i] = .5
        whole = indices.copy()
        parts = indices.copy()
        for i in Z_, Y_, X_:
            whole[i] = (indices[i] // true_factors[i]).astype(int)
            parts[i] = (indices[i] / true_factors[i] - whole[i]).astype(float)
    del indices

    # Trilinear interpolation determines the value of a new pixel by
    # comparing the values of the eight old pixels that surround it.
    # The hashes are the keys to the dictionary that contains those
    # old pixel values. The key indicates the position of the pixel
    # on each axis, with one meaning the position is ahead of the
    # new pixel, and zero meaning the position is behind it.
    hashes = [f'{n:>03b}'[::-1] for n in range(2 ** 3)]
    hash_table = {}

    # The original array needs to be made one dimensional for the
    # numpy.take operation that will occur as we build the tables.
    raveled = np.ravel(a)

    # Build the table that contains the old pixel values to
    # interpolate.
    for hash in hashes:
        hash_whole = whole.copy()

        # Use the hash key to adjust the which old pixel we are
        # looking at.
        for axis in Z_, Y_, X_:
            if hash[axis] == '1':
                hash_whole[axis] += 1

                # Handle the pixels that were pushed off the far
                # edge of the original array by giving them the
                # value of the last pixel along that axis in the
                # original array.
                m = np.zeros(hash_whole[axis].shape, dtype=bool)
                m[hash_whole[axis] >= a.shape[axis]] = True
                hash_whole[axis][m] = a.shape[axis] - 1

        # Since numpy.take() only works in one dimension, we need to
        # map the three dimensional indices of the original array to
        # the one dimensional indices used by the raveled version of
        # that array.
        raveled_indices = hash_whole[Z_] * a.shape[Y_] * a.shape[X_]
        raveled_indices += hash_whole[Y_] * a.shape[X_]
        raveled_indices += hash_whole[X_]

        # Get the value of the pixel in the original array.
        hash_table[hash] = np.take(raveled, raveled_indices.astype(int))

    # Once the hash table has been built, clean up the working arrays
    # in case we are running short on memory.
    else:
        del hash_whole, raveled_indices, whole

    # Everything before this was to set up the interpolation. Now that
    # it's set up, we perform the interpolation. Since we are doing
    # this across three dimensions, it's a three stage process. Stage
    # one is along the X_ axis.
    x1 = lp.lerp(hash_table['000'], hash_table['001'], parts[X_])
    x2 = lp.lerp(hash_table['010'], hash_table['011'], parts[X_])
    x3 = lp.lerp(hash_table['100'], hash_table['101'], parts[X_])
    x4 = lp.lerp(hash_table['110'], hash_table['111'], parts[X_])

    # Stage two is along the Y_ axis.
    y1 = lp.lerp(x1, x2, parts[Y_])
    y2 = lp.lerp(x3, x4, parts[Y_])
    del x1, x2, x3, x4

    # And stage three is along the Z_ axis. Since this is the last step
    # we can just return the result.
    return lp.lerp(y1, y2, parts[Z_])


# Private functions.
def _build_relative_position_masks(dimensions: int) -> list[str]:
    """Create the masks for identifying the different points used in
    an n-dimensional interpolation.
    """
    points = range(2 ** dimensions)
    mask_template = '{:>0' + str(dimensions) + 'b}'
    mask = [mask_template.format(p)[::-1] for p in points]
    return sorted(mask)


def _calc_raveled_indices(
    indices: IntAry,
    src_shape: Size
) -> IntAry:
    """Convert indices for a multidimensional array to indices for
    that array after it has been raveled.
    """
    axes = range(len(src_shape))
    raveled_shape = (indices.shape[0], *indices.shape[2:])
    raveled_indices = np.zeros(raveled_shape, dtype=int)
    for axis in axes:
        remaining_dims = src_shape[axis + 1:]
        axis_mod = prod(remaining_dims)
        raveled_indices += indices[:, axis] * axis_mod
    return raveled_indices


def _get_resizing_factors(
    src_shape: Size,
    dst_shape: Size
) -> tuple[float, ...]:
    """Determine how much each axis is resized by."""
    # The ternary is a quick fix for cases where there are dimensions
    # of length one. It may cause weird effects, so a more thoughtful
    # fix would be good in the future.
    src_ends = [n - 1 if n != 1 else 1 for n in src_shape]
    dst_ends = [n - 1 if n != 1 else 1 for n in dst_shape]
    factors = tuple(d / s for s, d in zip(src_ends, dst_ends))
    return factors


def _map_indices_and_distances(
    shape: Size,
    factors: tuple[float, ...]
) -> tuple[RatioAry, RatioAry]:
    """Map the indices for the zero position array and the distances
    for the distance array for an array resizing interpolation.
    """
    axes = range(len(shape))
    indices = np.indices(shape, dtype=float)
    for axis in axes:
        indices[axis] /= factors[axis]
    src_indices = np.trunc(indices)
    distances = indices - src_indices
    return src_indices, distances


def _replace_indices_with_values(
    src: NumAry,
    indices: IntAry
) -> NumAry:
    """Replace the indices in an array with values from another array."""
    # numpy.take only works in one dimension. We'll need to
    # ravel the original array to be able to get the values, but
    # we still need the original shape to calculate the new indices.
    src_shape = src.shape
    raveled = np.ravel(src)

    # Calculate the raveled indices for each dimension.
    raveled_indices = _calc_raveled_indices(indices, src_shape)

    # Return the values from the original array.
    result = np.take(raveled, raveled_indices.astype(int))
    return result

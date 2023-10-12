"""
maze
====

A pseudorandomly generated maze.

.. autoclass:: pjimg.imggen.Maze
.. autoclass:: pjimg.imggen.AnimatedMaze
.. autoclass:: pjimg.imggen.SolvedMaze

"""
from operator import itemgetter
from typing import Any, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from pjimg.imggen import unitnoise as un
from pjimg.util import ImgAry, IntAry, IntAry64, Loc, Size, X, Y, Z


# Types.
Spot = tuple[int, ...]
Step = tuple[Spot, Spot]
MazePath = list[Step]


# Public classes.
class Maze(un.UnitNoise):
    """A class to generate maze-like paths.

    .. figure:: images/maze.jpg
       :alt: A picture of an image created from the output of
            :class:`Maze`.
       
       Output of :class:`Maze`.
    
    :param unit: The number of pixels between vertices along an
        axis. The vertices are the locations where the direction
        of the path can change.
    :param width: (Optional.) The width of the path. This is the
        percentage of the width of the X axis length of the size
        of the fill. Values over one will probably be weird, but
        not in a great way.
    :param inset: (Optional.) Sets how many units from the end of
        the image to draw the path. Units here refers to the unit
        parameter from the UnitNoise parent class.
    :param origin: (Optional.) Where in the grid to start the path.
        This can be either a descriptive string or a three-dimensional
        coordinate. It defaults to the top-left corner of the first
        three-dimensional slice of the data.
    :param min: (Optional.) The minimum value of a vertex of the unit
        grid. This is involved in setting the path through the maze.
    :param max: (Optional.) The maximum value of a vertex of the unit
        grid. This is involved in setting the path through the maze.
    :param repeats: (Optional.) The number of times each value can
        appear on the unit grid. This is involved in setting the
        maximum size of noise that can be generated from the object.
    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not product the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :return: :class:Maze object.
    :rtype: imggen.maze.Maze

    Descriptive Origins
    -------------------
    The origin parameter can accept a description of the location
    instead of direct coordinates. This string must either be two
    words delimited by a hyphen or two letters. The first position
    sets the Y axis location can be one of the following options:

    *   top | t
    *   middle | m
    *   bottom | b

    The second position sets the X axis position and can be one of
    the following options:

    *   left | l
    *   middle | m
    *   right | r
    """
    def __init__(
        self, unit: Sequence[int],
        width: float = .2,
        inset: Sequence[int] = (0, 1, 1),
        origin: Union[str, Sequence[int]] = 'tl',
        min: int = 0x00,
        max: int = 0xff,
        repeats: int = 1,
        seed: un.Seed = None
    ) -> None:
        """Initialize an instance of Maze."""
        super().__init__(unit, min, max, repeats, seed)
        self.width = width
        self.inset = inset
        self.origin = origin

    # Public methods.
    def fill(
        self, size: Size,
        loc: Loc = (0, 0, 0)
    ) -> ImgAry:
        """Fill a volume with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`numpy.ndarray` with image data.
        :rtype: numpy.ndarray
        """
        values, unit_dim = self._build_grid(size, loc)
        path = self._build_path(values, unit_dim)
        # import pprint
        # pprint.pprint(path)
        return self._draw_path(path, size)

    # Private methods.
    def _build_grid(
        self, size: Size, loc: Loc
    ) -> tuple[IntAry64, Sequence[int]]:
        """Create a grid of values. This uses the same technique
        Perlin noise uses to add randomness to the noise. A table of
        values was shuffled, and we use the coordinate of each vertex
        within the grid as part of the process to lookup the table
        value for that vertex. This grid will be used to determine the
        route the path follows through the space.
        """
        unit_dim = tuple(int(s / u) for s, u in zip(size, self.unit))
        unit_dim = tuple(np.array(unit_dim) + np.array((0, 1, 1)))
        unit_dim = tuple(np.array(unit_dim) - np.array(self.inset) * 2)
        unit_indices = np.indices(unit_dim)
        for axis in X, Y:
            unit_indices[axis] += loc[axis]
        unit_indices[Z].fill(loc[Z])
        values = np.take(self._table, unit_indices[X])
        values += unit_indices[Y]
        values = np.take(self._table, values % len(self._table))
        values += unit_indices[Z]
        values = np.take(self._table, values & len(self._table))
        return values, unit_dim

    def _build_path(
        self, values: IntAry64,
        unit_dim: Sequence[int]
    ) -> MazePath:
        """Create the steps in the path."""
        # The cursor will be used to determine our current position
        # on the grid as we create the path.
        cursor = self._calc_origin(self.origin, unit_dim)

        # This will be used to track the grid vertices we've already
        # been to as we create the path. It allows us to keep the
        # path from looping back into itself.
        been_there = np.zeros(unit_dim, bool)
        been_there[tuple(cursor)] = True

        # These are the positions of the vertices the cursor could
        # move to next as it creates the path.
        vertices = np.array([
            (0, 0, -1),
            (0, 0, 1),
            (0, -1, 0),
            (0, 1, 0),
        ])

        # The index tracks where we are along the path. This is used
        # to allow us to go back up the path and create a new branch
        # if we run into a dead end while creating the path. It also
        # is how we know we're done when creating the path.
        index = 0

        # Create the path.
        path = []
        while True:

            # Look at the options available for the direction the path
            # can take. Some of them won't be viable because they are
            # outside the bounds of the image or have already been
            # hit.
            acursor = np.array(cursor)
            options = [vertex + acursor for vertex in vertices]
            viable = [
                (o, values[tuple(o)]) for o in options
                if self._is_viable_option(o, unit_dim, been_there)
            ]

            # If there is a viable next step, take that step.
            if viable:
                cursor = tuple(acursor)
                viable = sorted(viable, key=itemgetter(1))
                newloc: Spot = tuple(viable[0][0])
                path.append((cursor, newloc))
                been_there[newloc] = True
                cursor = newloc
                index = len(path)

            # If there is not a viable next step, go back to the last
            # place you were, so to see if there are any viable steps
            # there. If this goes all the way back to the beginning
            # of the path and there are no viable paths, then the
            # path is complete.
            else:
                index -= 1
                if index < 0:
                    break
                cursor = path[index][0]

        return path

    def _calc_origin(
        self, origin: Union[str, Sequence[int]],
        unit_dim: Sequence[int]
    ) -> Spot:
        "Determine the starting location of the cursor."
        # If origin isn't a string, no further calculation is needed.
        if not isinstance(origin, str):
            return tuple(origin)

        # Coordinates serialized as strings should be comma delimited.
        if ',' in origin:
            parts = origin.split(',')
            return tuple(int(part.strip()) for part in parts)

        # If it's neither of the above, it's a descriptive string.
        result = [0, 0, 0]
        items: Union[str, Sequence[str]] = origin
        if isinstance(items, str) and '-' in items:
            items = items.split('-')

        # Allow middle to be a shortcut for middle-middle.
        if items == 'middle' or items == 'm':
            items = 'mm'

        # Set the Y axis coordinate.
        if items[0] in ('top', 't'):
            result[Y] = 0
        if items[0] in ('middle', 'm'):
            result[Y] = unit_dim[Y] // 2
        if items[0] in ('bottom', 'b'):
            result[Y] = unit_dim[Y] - 1

        # Set the X axis coordinate.
        if items[1] in ('left', 'l'):
            result[X] = 0
        if items[1] in ('middle', 'm'):
            result[X] = unit_dim[X] // 2
        if items[1] in ('right', 'r'):
            result[X] = unit_dim[X] - 1

        return tuple(result)

    def _draw_path(
        self, path: MazePath,
        size: Size
    ) -> ImgAry:
        """Turn the unit grid array into an array of image data."""
        a = np.zeros(size, dtype=float)
        width = int(self.unit[-1] * self.width)
        for step in path:
            start = self._unit_to_pixel(step[0])
            end = self._unit_to_pixel(step[1])
            slice_y = self._get_slice(start[Y], end[Y], width)
            slice_x = self._get_slice(start[X], end[X], width)
            a[:, slice_y, slice_x] = 1.0
        return a

    def _get_slice(self, start: int, end: int, width: int) -> slice:
        """Get a slice of the array of image data of the given width."""
        if start > end:
            start, end = end, start
        start -= width
        end += width
        return slice(start, end)

    def _is_viable_option(
        self, option: IntAry,
        unit_dim: Sequence[int],
        been_there: NDArray[np.bool_]
    ) -> bool:
        loc = tuple(option)
        if (
            np.min(option) >= 0
            and all(unit_dim > option)
            and not been_there[loc]
        ):
            return True
        return False

    def _unit_to_pixel(self, unit_loc: Sequence[int]) -> Sequence[int]:
        """Convert an index of the unit grid array into an index
        of the image data.
        """
        unit = np.array(self.unit)
        pixel_loc = np.array(unit_loc) * unit
        pixel_loc += np.array(self.inset) * unit
        return tuple(pixel_loc)


class AnimatedMaze(Maze):
    """Animate the creation of a maze.

    :param unit: The number of pixels between vertices along an
        axis. The vertices are the locations where the direction
        of the path can change.
    :param delay: (Optional.) The number of frames to wait before
        starting the animation.
    :param linger: (Optional.) The number of frames to hold on the
        last image of the animation.
    :param trace: (Optional.) Whether to show all of the path that
        had been walked to this point (True) or just show this step
        (False).
    :param width: (Optional.) The width of the path. This is the
        percentage of the width of the X axis length of the size
        of the fill. Values over one will probably be weird, but
        not in a great way.
    :param inset: (Optional.) Sets how many units from the end of
        the image to draw the path. Units here refers to the unit
        parameter from the UnitNoise parent class.
    :param origin: (Optional.) Where in the grid to start the path.
        This can be either a descriptive string or a three-dimensional
        coordinate. It defaults to the top-left corner of the first
        three-dimensional slice of the data.
    :param min: (Optional.) The minimum value of a vertex of the unit
        grid. This is involved in setting the path through the maze.
    :param max: (Optional.) The maximum value of a vertex of the unit
        grid. This is involved in setting the path through the maze.
    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not product the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :return: :class:AnimatedMaze object.
    :rtype: imggen.maze.AnimatedMaze
    """
    def __init__(
        self, unit: Sequence[int],
        delay: int = 0,
        linger: int = 0,
        trace: bool = True,
        width: float = .2,
        inset: Sequence[int] = (0, 1, 1),
        origin: Union[str, Sequence[int]] = 'tl',
        min: int = 0x00,
        max: int = 0xff,
        repeats: int = 1,
        seed: un.Seed = None
    ) -> None:
        self.delay = delay
        self.linger = linger
        self.trace = trace
        super().__init__(unit, width, inset, origin, min, max, repeats, seed)

    # Public methods.
    def fill(
        self, size: Size,
        loc: Loc = (0, 0, 0)
    ) -> ImgAry:
        """Fill a volume with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`numpy.ndarray` with image data.
        :rtype: numpy.ndarray
        """
        a = super().fill(size, loc)
        for _ in range(self.delay):
            a = np.insert(a, 0, np.zeros_like(a[0]), 0)
        for _ in range(self.linger):
            a = np.insert(a, -1, a[-1], 0)
        return a

    # Private methods.
    def _draw_path(self, path: MazePath, size: Size) -> ImgAry:
        def _take_step(branch, frame):
            try:
                step = branch[index]
                start = self._unit_to_pixel(step[0])
                end = self._unit_to_pixel(step[1])
                slice_y = self._get_slice(start[Y], end[Y], width)
                slice_x = self._get_slice(start[X], end[X], width)
                frame[slice_y, slice_x] = 1.0
            except IndexError:
                pass
            except TypeError:
                pass
            return frame

        a = np.zeros(size, dtype=float)
        branches = self._find_branches(path)
        width = int(self.unit[-1] * self.width)
        index = 0
        frame = a[0].copy()
        while index < size[Z] - 1:
            for branch in branches:
                frame = _take_step(branch, frame)
            a[index + 1] = frame.copy()
            index += 1
            if not self.trace:
                frame.fill(0)
        return a

    def _find_branches(self, path: MazePath) -> list[list[Optional[Step]]]:
        """Find the spots where the path starts from the same location
        and split those out into branches, so they can be animated to
        be walked at the same time.
        """
        branches = []
        index = 1
        starts = [step[0] for step in path]
        branch: list[Optional[Step]] = [path[0],]

        # Trace all the branches in the maze.
        while index < len(path):
            start = path[index][0]
            if start in starts[:index]:
                branches.append(branch)
                for item in branches:
                    bstarts: list[Optional[Spot]] = []
                    for step in item:
                        if step:
                            bstarts.append(step[0])
                        else:
                            bstarts.append(step)
                    if start in bstarts:
                        delay = bstarts.index(start) - 1
                        branch = [None for _ in range(delay)]
                        break
                else:
                    msg = "Couldn't find branch with start."
                    raise ValueError(msg)
            branch.append(path[index])
            index += 1

        # Make sure the last branch we were working on gets counted.
        branches.append(branch)

        # Make sure all the branches are the same length.
        biggest = max(len(branch) for branch in branches)
        for branch in branches:
            if len(branch) < biggest:
                branch.append(None)
        return branches


class SolvedMaze(Maze):
    """Draw a line that shows how to get from one location to another
    in a maze.

    .. figure:: images/solvedmaze.jpg
       :alt: A picture of an image created from the output of
            :class:`SolvedMaze`.
       
       Output of :class:`SolvedMaze`.
    
    :param unit: The number of pixels between vertices along an
        axis. The vertices are the locations where the direction
        of the path can change.
    :param start: (Optional.) The starting location of the path.
        This can be either a descriptive string or a three-dimensional
        coordinate. It defaults to the top-left corner of the first
        three-dimensional slice of the data.
    :param end: (Optional.) The ending location of the path.
        This can be either a descriptive string or a three-dimensional
        coordinate. It defaults to the top-left corner of the first
        three-dimensional slice of the data.
    :param width: (Optional.) The width of the path. This is the
        percentage of the width of the X axis length of the size
        of the fill. Values over one will probably be weird, but
        not in a great way.
    :param inset: (Optional.) Sets how many units from the end of
        the image to draw the path. Units here refers to the unit
        parameter from the UnitNoise parent class.
    :param origin: (Optional.) Where in the grid to start the path.
        This can be either a descriptive string or a three-dimensional
        coordinate. It defaults to the top-left corner of the first
        three-dimensional slice of the data.
    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not product the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :return: :class:SolvedMaze object.
    :rtype: imggen.maze.SolvedPath

    Descriptive Origins
    -------------------
    The origin parameter can accept a description of the location
    instead of direct coordinates. This string must either be two
    words delimited by a hyphen or two letters. The first position
    sets the Y axis location can be one of the following options:

    *   top | t
    *   middle | m
    *   bottom | b

    The second position sets the X axis position and can be one of
    the following options:

    *   left | l
    *   middle | m
    *   right | r
    """
    def __init__(
        self, unit: Sequence[int],
        start: Union[str, Sequence[int]] = 'tl',
        end: Union[str, Sequence[int]] = 'br',
        algorithm: str = 'branches',
        width: float = .2,
        inset: Sequence[int] = (0, 1, 1),
        origin: Union[str, Sequence[int]] = 'tl',
        min: int = 0x00,
        max: int = 0xff,
        repeats: int = 1,
        seed: un.Seed = None
    ) -> None:
        super().__init__(unit, width, inset, origin, min, max, repeats, seed)
        self.start = start
        self.end = end
        self.algorithm = algorithm
        self._solve_path = self._solve_path_branches
        if algorithm == 'breadcrumb':
            self._solve_path = self._solve_path_breadcrumbs

    # Properties.
    @property
    def algorithm(self) -> str:
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value: str) -> None:
        self._solve_path = self._solve_path_branches
        if value == 'breadcrumb':
            self._solve_path = self._solve_path_breadcrumbs
        self._algorithm: str = value

    # Public methods.
    def fill(
        self, size: Size,
        loc: Loc = (0, 0, 0)
    ) -> ImgAry:
        """Fill a volume with image data.

        :param size: The size of the volume of image data to generate.
        :param loc: (Optional.) How much to shift the starting point
            for the noise generation along each axis.
        :return: An :class:`numpy.ndarray` with image data.
        :rtype: numpy.ndarray
        """
        values, unit_dim = self._build_grid(size, loc)
        path = self._build_path(values, unit_dim)
        solution = self._solve_path(path, unit_dim)
        return self._draw_path(solution, size)

    # Private methods.
    def _map_available_steps(self, path: MazePath) -> dict[Spot, list[Spot]]:
        """For every location in the path, determine what other
        locations the cursor can move to.
        """
        # Enumerate the grid locations in the path and create a set
        # that contains the available locations that can be stepped
        # to. A set is used in this stage to ensure the list of
        # available next steps doesn't contain duplicates.
        steps = {}
        for step in path:
            if step[0] not in steps:
                steps[step[0]] = set([step[1],])
            else:
                steps[step[0]].add(step[1])

            if step[1] not in steps:
                steps[step[1]] = set([step[0],])
            else:
                steps[step[1]].add(step[0])

        # The sets are returned as lists to allow for future sorting.
        return {k: list(steps[k]) for k in steps}

    def _solve_path_breadcrumbs(
        self, path: MazePath,
        unit_dim: Sequence[int]
    ) -> MazePath:
        """Determine the steps needed to move from one location in the
        path to another.
        """
        steps = self._map_available_steps(path)
        solution: MazePath = []
        been_there = np.zeros(unit_dim, int)
        start = tuple(self._calc_origin(self.start, unit_dim))
        end = tuple(self._calc_origin(self.end, unit_dim))
        last = None

        # Starting at the start location, walk through the path one
        # step at a time until the cursor reaches the end location.
        cursor = start
        while cursor != end:

            # Drop breadcrumbs so the algorithm knows how many times
            # you've been to this position.
            been_there[cursor] += 1

            # Create list with each of the possible next locations.
            # Then determine how many times the cursor has been to
            # each of those locations. If a location has been visited
            # it means we either just left that location or there was
            # a dead end in that direction. Pick the step that has
            # been visited the least because it's more likely there is
            # unexplored portions of the path in that direction.
            options = steps[cursor]
            times_hit = [been_there[option] for option in options]
            sort_options = sorted(zip(times_hit, options))
            next_ = sort_options[0][1]

            # If the location that we've visited the least is the
            # location we just came from, we have hit a dead end.
            # Since we don't know where we went wrong, start back
            # at the beginning so we can use the breadcrumbs to
            # find a more promising route.
            if next_ == last:
                cursor = start
                last = None
                solution = []

            # Otherwise, add the step to the possible solution, make
            # sure we remember the last location, so we can detect
            # dead ends, and move to the next location.
            else:
                solution.append((cursor, next_))
                last = cursor
                cursor = next_

        # Return the list of steps to go from the start of the path
        # to the end of the path.
        return solution

    def _solve_path_branches(
        self, path: MazePath,
        unit_dim: Sequence[int]
    ) -> MazePath:
        """Determine the steps needed to move from one location in the
        path to another.
        """
        # Determine the maximum number of steps it could possibly
        # take to use to determine when the algorithm gets stuck
        # in a loop because there is no solution.
        max_steps = unit_dim[Y] * unit_dim[X]

        # Get a map of where you can go with one step from each
        # location in the grid
        available_steps = self._map_available_steps(path)

        # Calculate the starting and ending locations.
        start_ = tuple(self._calc_origin(self.start, unit_dim))
        end = tuple(self._calc_origin(self.end, unit_dim))

        # Prime the possible paths through the maze with the first
        # steps that can be taken from the starting position.
        paths = []
        for option in available_steps[start_]:
            step = (start_, option)
            path = [step,]
            paths.append(path)

        # Follow each path possible from the starting point, breaking
        # once one of the paths reaches the exit or the paths take
        # enough steps to have reached every position in the maze
        # without finding the exit.
        step_count = 0
        solution = None
        while not solution and step_count <= max_steps:
            new_paths = []
            for path in paths:
                step = path[-1]

                # Hurray, we found the exit!
                if step[1] == end:
                    solution = path
                    break

                # We haven't found the exit yet, so keep looking.
                options = available_steps[step[1]]
                options = [option for option in options if option != step[0]]
                for option in options:
                    new_path = path[:]
                    new_step = (step[1], option)
                    new_path.append(new_step)
                    new_paths.append(new_path)

            # Since we didn't find the exit, get ready for the next
            # iteration.
            paths = new_paths
            step_count += 1

        # If we took enough steps to reach every position in the
        # maze without finding an exit, there must not be a solution.
        if not solution:
            raise ValueError('No solution exists for path.')

        # Return the solution.
        return solution

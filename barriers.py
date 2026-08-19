"""
Obstacle layouts.

A layout is a function that returns a boolean grid the same shape as the world:
True means the cell is solid and nothing can enter it. Barriers are what turn
"head west" from a complete solution into one that strands a creature in a
dead end, so they are the cheapest way to make an objective genuinely hard.

Adding a layout is one function plus an entry in `LAYOUTS`. Everything else --
placement, movement, the `blocked_forward` sense, the animation -- picks it up
automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

type Layout = Callable[[int, int], npt.NDArray[np.bool_]]
"""Given (width, height), return the grid of solid cells."""


def none(width: int, height: int) -> npt.NDArray[np.bool_]:
    """An empty box."""
    return np.zeros((width, height), dtype=bool)


def wall(width: int, height: int) -> npt.NDArray[np.bool_]:
    """A single vertical wall with one gap in the middle of it."""
    grid = np.zeros((width, height), dtype=bool)
    x = width // 2
    gap = height // 10
    grid[x - 1 : x + 1, :] = True
    grid[x - 1 : x + 1, height // 2 - gap : height // 2 + gap] = False
    return grid


def slalom(width: int, height: int) -> npt.NDArray[np.bool_]:
    """
    Three walls with gaps at alternating ends.

    A straight dash west fails here: a creature has to move along a wall to
    find the opening, which is exactly what `blocked_forward` is good for.
    """
    grid = np.zeros((width, height), dtype=bool)
    gap = height // 4
    for i, x in enumerate((width // 4, width // 2, 3 * width // 4)):
        grid[x - 1 : x + 1, :] = True
        if i % 2 == 0:
            grid[x - 1 : x + 1, :gap] = False
        else:
            grid[x - 1 : x + 1, height - gap :] = False
    return grid


def pillars(width: int, height: int) -> npt.NDArray[np.bool_]:
    """A regular field of square blocks to weave between."""
    grid = np.zeros((width, height), dtype=bool)
    spacing = max(8, width // 8)
    size = max(2, spacing // 3)
    for x in range(spacing, width - spacing + 1, spacing):
        for y in range(spacing, height - spacing + 1, spacing):
            grid[x : x + size, y : y + size] = True
    return grid


def funnel(width: int, height: int) -> npt.NDArray[np.bool_]:
    """
    Two diagonal walls narrowing to a small opening on the west side.

    Crowding matters here -- the gap only fits a few creatures at a time, so
    the neighbour senses start to earn their keep.
    """
    grid = np.zeros((width, height), dtype=bool)
    mouth = max(2, height // 20)
    start, end = width // 3, 2 * width // 3
    for x in range(start, end):
        # Wide open at the eastern end, narrowing to `mouth` at the west.
        along = (x - start) / max(1, end - 1 - start)
        half_gap = int(mouth + along * (height / 2 - mouth))
        grid[x, : height // 2 - half_gap] = True
        grid[x, height // 2 + half_gap :] = True
    return grid


LAYOUTS: Final[dict[str, Layout]] = {
    "none": none,
    "wall": wall,
    "slalom": slalom,
    "pillars": pillars,
    "funnel": funnel,
}


def build(name: str, width: int, height: int) -> npt.NDArray[np.bool_]:
    """Build a named layout, raising a helpful error if the name is unknown."""
    try:
        layout = LAYOUTS[name]
    except KeyError:
        known = ", ".join(sorted(LAYOUTS))
        raise ValueError(f"unknown barrier layout {name!r}; try one of: {known}") from None
    return layout(width, height)

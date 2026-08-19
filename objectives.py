"""
What it takes to reproduce.

An objective is the entire selection pressure, and swapping it is the main dial
for changing what evolves. The old position-only criterion could express
"be here at the end" and nothing else; an `Objective` can also watch an
organism throughout the generation and change the world while it runs.

Three hooks, all optional except `survives`:

- `begin_generation`  set up, once per generation
- `advance`           move whatever the objective controls, once per timestep
- `observe`           watch one organism, once per timestep -- this is how an
                      objective accumulates history in `organism.progress`
- `survives`          the verdict, at the end of the generation

`zones` reports the regions worth drawing, so the animation illustrates any new
objective without being taught about it.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from organism import Organism, World

type Zone = Callable[[int, int, "World"], bool]
"""Whether a given cell belongs to a named region of the world."""


@dataclass(frozen=True, slots=True)
class Shading:
    """A region for the animation to draw."""

    mask: npt.NDArray[np.bool_]
    colour: str
    label: str


# ----------------------------------------------------------------------------
# Regions
# ----------------------------------------------------------------------------


def left_edge(x: int, y: int, world: World) -> bool:
    """A band along the west wall."""
    return x < world.width * world.zone_fraction


def right_edge(x: int, y: int, world: World) -> bool:
    """A band along the east wall."""
    return x >= world.width * (1 - world.zone_fraction)


def centre(x: int, y: int, world: World) -> bool:
    """A circle at the middle of the world."""
    radius = world.width * world.zone_fraction
    dx = x - world.width / 2
    dy = y - world.height / 2
    return dx * dx + dy * dy <= radius * radius


def top_edge(x: int, y: int, world: World) -> bool:
    """A band along the north wall."""
    return y >= world.height * (1 - world.zone_fraction)


def bottom_edge(x: int, y: int, world: World) -> bool:
    """A band along the south wall."""
    return y < world.height * world.zone_fraction


def corners(x: int, y: int, world: World) -> bool:
    """All four corners."""
    span = world.width * world.zone_fraction
    near_x = x < span or x >= world.width - span
    near_y = y < span or y >= world.height - span
    return near_x and near_y


def mask_of(zone: Zone, world: World) -> npt.NDArray[np.bool_]:
    """Evaluate a zone over every cell, so it can be drawn."""
    return np.array(
        [[zone(x, y, world) for y in range(world.height)] for x in range(world.width)],
        dtype=bool,
    )


# ----------------------------------------------------------------------------
# Objectives
# ----------------------------------------------------------------------------


class Objective(ABC):
    """The rule that decides who reproduces."""

    name: str = "objective"

    dynamic: bool = False
    """True if `zones` changes during a generation and must be redrawn."""

    # These three are deliberately optional: most objectives only need a
    # verdict, so doing nothing is the correct default rather than an oversight.
    def begin_generation(self, world: World) -> None:  # noqa: B027
        """Reset any state held for a whole generation."""

    def advance(self, world: World) -> None:  # noqa: B027
        """Move whatever this objective controls. Called once per timestep."""

    def observe(self, org: Organism, world: World) -> None:  # noqa: B027
        """Watch one organism for one timestep."""

    @abstractmethod
    def survives(self, org: Organism, world: World) -> bool:
        """Whether this organism gets to reproduce."""

    def zones(self, world: World) -> list[Shading]:
        """Regions worth drawing. Empty means the objective has nothing to show."""
        return []


class ReachZone(Objective):
    """
    Be inside a region when the generation ends.

    The original rule, and still the easiest to evolve against: only the final
    instant matters, so a creature can do anything it likes on the way.
    """

    def __init__(self, name: str, zone: Zone) -> None:
        self.name = name
        self.zone = zone

    def survives(self, org: Organism, world: World) -> bool:
        return self.zone(org.x, org.y, world)

    def zones(self, world: World) -> list[Shading]:
        return [Shading(mask_of(self.zone, world), "Greens", "reach by the end")]


class StayInZone(Objective):
    """
    Spend most of the generation inside a region, not merely end there.

    A far harsher rule than `ReachZone`: arriving late is worthless, so it
    rewards getting there fast and then holding position against the crowd.
    """

    def __init__(self, name: str, zone: Zone, fraction: float | None = None) -> None:
        self.name = name
        self.zone = zone
        self.fraction = fraction

    def _required(self, world: World) -> float:
        fraction = self.fraction if self.fraction is not None else world.config.stay_fraction
        return fraction * world.config.steps_per_generation

    def begin_generation(self, world: World) -> None:
        for org in world.organisms:
            org.progress["in_zone"] = 0.0

    def observe(self, org: Organism, world: World) -> None:
        if self.zone(org.x, org.y, world):
            org.progress["in_zone"] = org.progress.get("in_zone", 0.0) + 1.0

    def survives(self, org: Organism, world: World) -> bool:
        return org.progress.get("in_zone", 0.0) >= self._required(world)

    def zones(self, world: World) -> list[Shading]:
        return [Shading(mask_of(self.zone, world), "Greens", "stay here")]


class VisitInOrder(Objective):
    """
    Touch one region, then finish in another.

    This is the first objective that cannot be solved by a fixed heading. A
    creature has to change its mind partway through, which means routing the
    `age` sense -- or a recurrent inner neuron -- into its movement.
    """

    def __init__(self, name: str, first: Zone, second: Zone) -> None:
        self.name = name
        self.first = first
        self.second = second

    def begin_generation(self, world: World) -> None:
        for org in world.organisms:
            org.progress["visited"] = 0.0

    def observe(self, org: Organism, world: World) -> None:
        if self.first(org.x, org.y, world):
            org.progress["visited"] = 1.0

    def survives(self, org: Organism, world: World) -> bool:
        return bool(org.progress.get("visited")) and self.second(org.x, org.y, world)

    def zones(self, world: World) -> list[Shading]:
        return [
            Shading(mask_of(self.first, world), "Blues", "visit first"),
            Shading(mask_of(self.second, world), "Greens", "finish here"),
        ]


class Hazard(Objective):
    """
    Survive a roaming danger that kills whatever it touches.

    Unlike every other objective, this one changes the world as it runs, and
    death is immediate rather than judged at the end. Selection stops being
    about geometry and starts being about reacting to something that moves.
    """

    def __init__(self, name: str = "hazard") -> None:
        self.name = name
        self.dynamic = True
        self.x = 0.0
        self.y = 0.0

    def _place(self, world: World) -> None:
        """Walk the hazard around a circle inscribed in the world."""
        period = max(1, world.config.hazard_period)
        angle = 2 * math.pi * (world.step % period) / period
        self.x = world.width / 2 + math.cos(angle) * world.width / 4
        self.y = world.height / 2 + math.sin(angle) * world.height / 4

    def begin_generation(self, world: World) -> None:
        self._place(world)

    def advance(self, world: World) -> None:
        self._place(world)
        radius = world.config.hazard_radius
        for org in world.organisms:
            if org.alive and math.dist((org.x, org.y), (self.x, self.y)) <= radius:
                world.kill(org)

    def survives(self, org: Organism, world: World) -> bool:
        return org.alive

    def zones(self, world: World) -> list[Shading]:
        radius = world.config.hazard_radius
        xs = np.arange(world.width)[:, None]
        ys = np.arange(world.height)[None, :]
        caught = (xs - self.x) ** 2 + (ys - self.y) ** 2 <= radius**2
        return [Shading(caught, "Reds", "hazard")]


def _build() -> dict[str, Objective]:
    """The objectives offered on the command line and in the notebook."""
    listed: list[Objective] = [
        ReachZone("left", left_edge),
        ReachZone("right", right_edge),
        ReachZone("centre", centre),
        ReachZone("corners", corners),
        StayInZone("stay", left_edge),
        StayInZone("stay-centre", centre),
        VisitInOrder("there-and-back", right_edge, left_edge),
        VisitInOrder("top-to-bottom", top_edge, bottom_edge),
        Hazard(),
    ]
    return {objective.name: objective for objective in listed}


OBJECTIVES: Final[dict[str, Objective]] = _build()

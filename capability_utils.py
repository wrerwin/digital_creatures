"""
Defines what an organism is able to perceive and what it is able to do.

These two enums are the organism's interface to the world. A genome wires
sensors to actions (possibly via inner neurons), so adding a member here
immediately widens the space of behaviours evolution can discover -- no other
file needs to change.

Sensor functions take (organism, world) and return a float in [-1, 1], so that
no single sensor dominates the weighted sums inside a brain.
"""

from __future__ import annotations

import random
from enum import IntEnum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable

    from organism import Organism, World

# Radius, in cells, of the neighbourhood an organism can feel around itself.
DENSITY_RADIUS: Final = 4


class Sensor(IntEnum):
    """Everything an organism can perceive. Values index into sensor readings."""

    X_POSITION = 0
    Y_POSITION = 1
    BORDER_DISTANCE = 2
    POPULATION_DENSITY = 3
    BLOCKED_FORWARD = 4
    LAST_MOVE_X = 5
    LAST_MOVE_Y = 6
    AGE = 7
    RANDOM = 8
    BIAS = 9

    def __str__(self) -> str:
        return self.name.lower()


class Action(IntEnum):
    """
    Everything an organism can do. Values index into the brain's output levels.

    Action neurons do not move the organism directly. Each produces a level in
    [-1, 1] and `Organism.act` resolves the competing levels into one step.
    """

    MOVE_X = 0
    """Signed drive along x: positive is east, negative is west."""

    MOVE_Y = 1
    """Signed drive along y: positive is north, negative is south."""

    MOVE_FORWARD = 2
    """Keep going the way you last went."""

    MOVE_RANDOM = 3
    """Take an arbitrary step."""

    STAY = 4
    """Suppress movement."""

    def __str__(self) -> str:
        return self.name.lower()


type SensorFn = Callable[[Organism, World], float]


# ----------------------------------------------------------------------------
# Sensor implementations
# ----------------------------------------------------------------------------


def sense_x_position(org: Organism, world: World) -> float:
    """Where the organism sits along x: 0 at the west wall, 1 at the east."""
    return org.x / (world.width - 1)


def sense_y_position(org: Organism, world: World) -> float:
    """Where the organism sits along y: 0 at the south wall, 1 at the north."""
    return org.y / (world.height - 1)


def sense_border_distance(org: Organism, world: World) -> float:
    """0 when pressed against the nearest wall, 1 when in the middle of the world."""
    to_wall = min(org.x, org.y, world.width - 1 - org.x, world.height - 1 - org.y)
    half_span = min(world.width, world.height) / 2
    return min(to_wall / half_span, 1.0)


def sense_population_density(org: Organism, world: World) -> float:
    """Fraction of nearby cells that hold another organism."""
    return world.population_density(org.x, org.y, DENSITY_RADIUS)


def sense_blocked_forward(org: Organism, world: World) -> float:
    """1 if the cell the organism last moved toward is now a wall or another organism."""
    if org.last_dx == 0 and org.last_dy == 0:
        return 0.0
    ahead_x, ahead_y = org.x + org.last_dx, org.y + org.last_dy
    return 0.0 if world.can_move_to(ahead_x, ahead_y) else 1.0


def sense_last_move_x(org: Organism, world: World) -> float:
    """The x component of the organism's previous step."""
    return float(org.last_dx)


def sense_last_move_y(org: Organism, world: World) -> float:
    """The y component of the organism's previous step."""
    return float(org.last_dy)


def sense_age(org: Organism, world: World) -> float:
    """0 at the start of a generation, 1 at the end of it."""
    return org.age / world.config.steps_per_generation


def sense_random(org: Organism, world: World) -> float:
    """Noise, so evolution has a source of unpredictability to draw on."""
    return random.uniform(-1.0, 1.0)


def sense_bias(org: Organism, world: World) -> float:
    """Constant input, which lets a connection act as a fixed drive."""
    return 1.0


SENSOR_FUNCTIONS: Final[dict[Sensor, SensorFn]] = {
    Sensor.X_POSITION: sense_x_position,
    Sensor.Y_POSITION: sense_y_position,
    Sensor.BORDER_DISTANCE: sense_border_distance,
    Sensor.POPULATION_DENSITY: sense_population_density,
    Sensor.BLOCKED_FORWARD: sense_blocked_forward,
    Sensor.LAST_MOVE_X: sense_last_move_x,
    Sensor.LAST_MOVE_Y: sense_last_move_y,
    Sensor.AGE: sense_age,
    Sensor.RANDOM: sense_random,
    Sensor.BIAS: sense_bias,
}

# A missing entry would only surface once a genome happened to wire that sensor,
# which could be thousands of generations in. Fail at import instead.
assert SENSOR_FUNCTIONS.keys() == set(Sensor), "every Sensor needs an implementation"


def read_sensors(org: Organism, world: World) -> dict[Sensor, float]:
    """Evaluate every sensor for one organism. For inspection, not for the hot loop."""
    return {sensor: fn(org, world) for sensor, fn in SENSOR_FUNCTIONS.items()}

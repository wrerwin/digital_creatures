"""
Defines what an organism is able to perceive and what it is able to do.

These two enums are the organism's interface to the world. A genome wires
sensors to actions (possibly via inner neurons), so adding a member here
immediately widens the space of behaviours evolution can discover -- no other
file needs to change.

Sensor functions take (organism, world) and return a float in [-1, 1], so that
no single sensor dominates the weighted sums inside a brain.

The senses fall into four groups:

- where am I           position, distance to the walls
- what is around me    neighbours and obstacles, and which *direction* they lie in
- what can I smell     the pheromone layer, which is the only channel one
                       organism has for affecting what another perceives
- what was I doing     previous movement, age, noise, a constant
"""

from __future__ import annotations

import inspect
import random
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable

    from organism import Organism, World


class Category(StrEnum):
    """
    How a capability is presented to somebody configuring a run.

    Three groups, because they answer three different questions: what can it
    perceive of the world, what can it do in the world, and what does it know
    about itself. The last is where memory and self-knowledge live, and is the
    one most easily overlooked when picking a creature apart.
    """

    SENSING = "sensing"
    MOVING = "moving"
    INTELLIGENCE = "intelligence"


class Sensor(IntEnum):
    """Everything an organism can perceive. Values index into sensor readings."""

    X_POSITION = 0
    Y_POSITION = 1
    BORDER_DISTANCE = 2
    POPULATION_DENSITY = 3
    NEIGHBOURS_EAST = 4
    NEIGHBOURS_NORTH = 5
    NEAREST_NEIGHBOUR = 6
    BLOCKED_FORWARD = 7
    BLOCKED_LEFT = 8
    BLOCKED_RIGHT = 9
    PHEROMONE_HERE = 10
    PHEROMONE_EAST = 11
    PHEROMONE_NORTH = 12
    LAST_MOVE_X = 13
    LAST_MOVE_Y = 14
    AGE = 15
    ENERGY = 16
    RANDOM = 17
    BIAS = 18

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

    MOVE_LEFT = 3
    """Turn ninety degrees left of the last heading and go."""

    MOVE_RANDOM = 4
    """Take an arbitrary step."""

    STAY = 5
    """Suppress movement."""

    EMIT_PHEROMONE = 6
    """Lay scent on the current cell, for others (and yourself) to smell later."""

    def __str__(self) -> str:
        return self.name.lower()


type SensorFn = Callable[[Organism, World], float]


# ----------------------------------------------------------------------------
# Where am I
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


# ----------------------------------------------------------------------------
# What is around me
# ----------------------------------------------------------------------------


def sense_population_density(org: Organism, world: World) -> float:
    """How crowded it is nearby, with no indication of direction."""
    return world.population_density(org.x, org.y, world.config.sense_radius)


def sense_neighbours_east(org: Organism, world: World) -> float:
    """
    Which side holds more neighbours: +1 all east, -1 all west, 0 balanced.

    Paired with `neighbours_north`, this is what makes following, flocking and
    fleeing reachable at all -- density alone only says "it is crowded here".
    """
    return world.neighbour_gradient(org.x, org.y, world.config.sense_radius)[0]


def sense_neighbours_north(org: Organism, world: World) -> float:
    """Which side holds more neighbours: +1 all north, -1 all south, 0 balanced."""
    return world.neighbour_gradient(org.x, org.y, world.config.sense_radius)[1]


def sense_nearest_neighbour(org: Organism, world: World) -> float:
    """1 when another organism is adjacent, 0 when none is within sensing range."""
    return world.nearest_neighbour(org.x, org.y, world.config.sense_radius)


def sense_blocked_forward(org: Organism, world: World) -> float:
    """1 if the cell the organism last moved toward is a wall, barrier or organism."""
    return _blocked(org, world, quarter_turns=0)


def sense_blocked_left(org: Organism, world: World) -> float:
    """1 if the cell ninety degrees left of the current heading is blocked."""
    return _blocked(org, world, quarter_turns=1)


def sense_blocked_right(org: Organism, world: World) -> float:
    """1 if the cell ninety degrees right of the current heading is blocked."""
    return _blocked(org, world, quarter_turns=-1)


def _blocked(org: Organism, world: World, quarter_turns: int) -> float:
    """Whether the cell that many quarter-turns left of the heading is unavailable."""
    dx, dy = org.last_dx, org.last_dy
    if dx == 0 and dy == 0:
        return 0.0
    # Rotating a quarter turn to the left maps (dx, dy) -> (-dy, dx).
    for _ in range(quarter_turns % 4):
        dx, dy = -dy, dx
    return 0.0 if world.can_move_to(org.x + dx, org.y + dy) else 1.0


# ----------------------------------------------------------------------------
# What can I smell
# ----------------------------------------------------------------------------


def sense_pheromone_here(org: Organism, world: World) -> float:
    """Strength of the scent on the organism's own cell."""
    return world.pheromone_at(org.x, org.y)


def sense_pheromone_east(org: Organism, world: World) -> float:
    """Which side smells stronger: +1 east, -1 west, 0 balanced or unscented."""
    return world.pheromone_gradient(org.x, org.y, world.config.sense_radius)[0]


def sense_pheromone_north(org: Organism, world: World) -> float:
    """Which side smells stronger: +1 north, -1 south, 0 balanced or unscented."""
    return world.pheromone_gradient(org.x, org.y, world.config.sense_radius)[1]


# ----------------------------------------------------------------------------
# What was I doing
# ----------------------------------------------------------------------------


def sense_last_move_x(org: Organism, world: World) -> float:
    """The x component of the organism's previous step."""
    return float(org.last_dx)


def sense_last_move_y(org: Organism, world: World) -> float:
    """The y component of the organism's previous step."""
    return float(org.last_dy)


def sense_age(org: Organism, world: World) -> float:
    """0 at the start of a generation, 1 at the end of it."""
    return org.age / world.config.steps_per_generation


def sense_energy(org: Organism, world: World) -> float:
    """How full the tank is: 1 at the start of a generation, 0 at starvation."""
    if org.config.initial_energy <= 0:
        return 0.0
    return max(0.0, min(1.0, org.energy / org.config.initial_energy))


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
    Sensor.NEIGHBOURS_EAST: sense_neighbours_east,
    Sensor.NEIGHBOURS_NORTH: sense_neighbours_north,
    Sensor.NEAREST_NEIGHBOUR: sense_nearest_neighbour,
    Sensor.BLOCKED_FORWARD: sense_blocked_forward,
    Sensor.BLOCKED_LEFT: sense_blocked_left,
    Sensor.BLOCKED_RIGHT: sense_blocked_right,
    Sensor.PHEROMONE_HERE: sense_pheromone_here,
    Sensor.PHEROMONE_EAST: sense_pheromone_east,
    Sensor.PHEROMONE_NORTH: sense_pheromone_north,
    Sensor.LAST_MOVE_X: sense_last_move_x,
    Sensor.LAST_MOVE_Y: sense_last_move_y,
    Sensor.AGE: sense_age,
    Sensor.ENERGY: sense_energy,
    Sensor.RANDOM: sense_random,
    Sensor.BIAS: sense_bias,
}

# A missing entry would only surface once a genome happened to wire that sensor,
# which could be thousands of generations in. Fail at import instead.
assert SENSOR_FUNCTIONS.keys() == set(Sensor), "every Sensor needs an implementation"


def read_sensors(org: Organism, world: World) -> dict[Sensor, float]:
    """Evaluate every sensor for one organism. For inspection, not for the hot loop."""
    return {sensor: fn(org, world) for sensor, fn in SENSOR_FUNCTIONS.items()}


# ----------------------------------------------------------------------------
# Presentation: how capabilities are grouped and explained
# ----------------------------------------------------------------------------
#
# Sensing is what a creature perceives of the world; intelligence is what it
# knows about itself, which is where memory and timing come from.

SENSOR_CATEGORY: Final[dict[Sensor, Category]] = {
    Sensor.X_POSITION: Category.SENSING,
    Sensor.Y_POSITION: Category.SENSING,
    Sensor.BORDER_DISTANCE: Category.SENSING,
    Sensor.POPULATION_DENSITY: Category.SENSING,
    Sensor.NEIGHBOURS_EAST: Category.SENSING,
    Sensor.NEIGHBOURS_NORTH: Category.SENSING,
    Sensor.NEAREST_NEIGHBOUR: Category.SENSING,
    Sensor.BLOCKED_FORWARD: Category.SENSING,
    Sensor.BLOCKED_LEFT: Category.SENSING,
    Sensor.BLOCKED_RIGHT: Category.SENSING,
    Sensor.PHEROMONE_HERE: Category.SENSING,
    Sensor.PHEROMONE_EAST: Category.SENSING,
    Sensor.PHEROMONE_NORTH: Category.SENSING,
    Sensor.LAST_MOVE_X: Category.INTELLIGENCE,
    Sensor.LAST_MOVE_Y: Category.INTELLIGENCE,
    Sensor.AGE: Category.INTELLIGENCE,
    Sensor.ENERGY: Category.INTELLIGENCE,
    Sensor.RANDOM: Category.INTELLIGENCE,
    Sensor.BIAS: Category.INTELLIGENCE,
}

assert SENSOR_CATEGORY.keys() == set(Sensor), "every Sensor needs a category"

ACTION_DESCRIPTIONS: Final[dict[Action, str]] = {
    Action.MOVE_X: "Drive east or west. Positive goes east, negative west.",
    Action.MOVE_Y: "Drive north or south. Positive goes north, negative south.",
    Action.MOVE_FORWARD: "Keep going whichever way it last moved.",
    Action.MOVE_LEFT: "Turn a quarter-turn left of its heading and go.",
    Action.MOVE_RANDOM: "Step in an arbitrary direction.",
    Action.STAY: "Damp whatever the other actions wanted, and hold still.",
    Action.EMIT_PHEROMONE: (
        "Lay scent on the current cell. The only way one creature can change "
        "what another perceives."
    ),
}

assert ACTION_DESCRIPTIONS.keys() == set(Action), "every Action needs a description"


def describe(capability: Sensor | Action) -> str:
    """
    A one-line explanation, for a tooltip.

    Sensor text comes from the sensing function's own docstring, so the
    explanation shown to a user cannot drift away from what the code does.
    """
    if isinstance(capability, Action):
        return ACTION_DESCRIPTIONS[capability]

    doc = inspect.getdoc(SENSOR_FUNCTIONS[capability]) or ""
    return " ".join(doc.split("\n\n")[0].split())


def catalogue() -> list[dict[str, object]]:
    """
    Every capability with the category and explanation the UI needs.

    Built from the enums themselves, so a new sense or action appears in the
    interface, in the right group, with no front-end change.
    """
    entries: list[dict[str, object]] = [
        {
            "value": int(sensor),
            "label": str(sensor),
            "kind": "sensors",
            "category": str(SENSOR_CATEGORY[sensor]),
            "description": describe(sensor),
        }
        for sensor in Sensor
    ]
    entries += [
        {
            "value": int(action),
            "label": str(action),
            "kind": "actions",
            "category": str(Category.MOVING),
            "description": describe(action),
        }
        for action in Action
    ]
    return entries

"""
The organisms and the world they live in.

A generation runs for `Settings.steps_per_generation` timesteps. At the end of
it a survival criterion decides who reproduces; everybody else dies. Survivors
are cloned with mutation until the population is full again, and the next
generation starts from scratch on an empty grid.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Final

import numpy as np

import brain_utils
import settings
from capability_utils import Action, Sensor, read_sensors

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import numpy.typing as npt

    from brain_utils import Genome
    from settings import Settings

# The eight ways a step can point, used when an organism moves at random.
DIRECTIONS: Final = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)  # fmt: skip

type Criterion = Callable[[int, int, "World"], bool]
"""Given a position and the world, decide whether an organism there survives."""


class Organism:
    """
    One creature: a position on the grid, a genome, and the brain it builds.

    An organism does not know how to find anything out for itself -- every
    sense it has comes from `capability_utils`, and which of those senses it
    actually consults is decided by its genome.
    """

    __slots__ = ("age", "brain", "config", "genome", "last_dx", "last_dy", "name", "x", "y")

    def __init__(
        self,
        config: Settings,
        genome: Genome | None = None,
        name: str | None = None,
    ) -> None:
        self.config = config
        self.name = name
        self.genome = genome if genome is not None else brain_utils.random_genome(config)
        self.brain = brain_utils.Brain(self.genome, config)

        # Filled in by World.place; an unplaced organism has no position.
        self.x: int = -1
        self.y: int = -1

        self.last_dx = 0
        self.last_dy = 0
        self.age = 0

    @property
    def placed(self) -> bool:
        """Whether this organism has been given a cell on the grid."""
        return self.x >= 0 and self.y >= 0

    # ------------------------------------------------------------------
    # Perceiving and acting
    # ------------------------------------------------------------------

    def perceive(self, world: World) -> dict[Sensor, float]:
        """
        Every sense this organism has, as a `Sensor` -> value mapping.

        This is for inspection and debugging. The brain reads sensors lazily
        during `think`, evaluating only the ones its genome refers to.
        """
        return read_sensors(self, world)

    def act(self, world: World) -> None:
        """
        Run the brain for one timestep and resolve its output into a step.

        The action neurons compete rather than take turns: their levels are
        summed into an urge along each axis, which is then converted into an
        actual grid step probabilistically, so a weak urge moves the organism
        sometimes and a strong one moves it almost always.
        """
        levels = self.brain.think(self, world)

        urge_x = levels[Action.MOVE_X]
        urge_y = levels[Action.MOVE_Y]

        forward = levels[Action.MOVE_FORWARD]
        urge_x += forward * self.last_dx
        urge_y += forward * self.last_dy

        if wander := levels[Action.MOVE_RANDOM]:
            dx, dy = random.choice(DIRECTIONS)
            urge_x += wander * dx
            urge_y += wander * dy

        # A positive 'stay' level damps whatever the other neurons wanted.
        stillness = max(0.0, levels[Action.STAY])
        urge_x *= 1.0 - stillness
        urge_y *= 1.0 - stillness

        self.move(_urge_to_step(urge_x), _urge_to_step(urge_y), world)
        self.age += 1

    def move(self, dx: int, dy: int, world: World) -> None:
        """
        Try to step by (dx, dy), refusing moves into walls or occupied cells.

        A blocked diagonal falls back to whichever single axis is still open,
        which lets organisms slide along a wall instead of sticking to it.
        """
        if dx == 0 and dy == 0:
            self.last_dx = self.last_dy = 0
            return

        for step_x, step_y in ((dx, dy), (dx, 0), (0, dy)):
            if step_x == 0 and step_y == 0:
                continue
            if world.can_move_to(self.x + step_x, self.y + step_y):
                world.relocate(self, self.x + step_x, self.y + step_y)
                self.last_dx, self.last_dy = step_x, step_y
                return

        # Fully boxed in: keep the heading so BLOCKED_FORWARD can report it.
        self.last_dx, self.last_dy = dx, dy

    # ------------------------------------------------------------------
    # Reproduction
    # ------------------------------------------------------------------

    def reproduce(self) -> Organism:
        """
        Produce one offspring: this organism's genome, copied with mutations.

        Reproduction is asexual, so all new variation comes from mutation. The
        child is unplaced -- the world decides where it starts.
        """
        return Organism(self.config, genome=brain_utils.mutate(self.genome, self.config))

    def __repr__(self) -> str:
        return f"<Organism at ({self.x}, {self.y})>"


def _urge_to_step(urge: float) -> int:
    """
    Turn a continuous urge into one of -1, 0, +1.

    The magnitude is read as a probability, so behaviour stays stochastic: an
    urge of 0.3 produces a step about a third of the time.
    """
    urge = max(-1.0, min(1.0, urge))
    if random.random() >= abs(urge):
        return 0
    return 1 if urge > 0 else -1


class World:
    """
    The grid, the population living on it, and the generational cycle.

    Occupancy is kept in a numpy array alongside the organism list: the list is
    what we iterate over, the array is what makes "is that cell taken?" and
    "how crowded is it here?" cheap enough to ask on every timestep.
    """

    def __init__(
        self,
        config: Settings | None = None,
        criterion: Criterion | None = None,
        n_organisms: int | None = None,
    ) -> None:
        self.config = config or settings.DEFAULT
        self.criterion = criterion or left_edge
        self.n_organisms = n_organisms if n_organisms is not None else self.config.n_organisms

        self.width = self.config.width
        self.height = self.config.height

        self.generation = 0
        self.step = 0
        self.occupancy: npt.NDArray[np.bool_] = np.zeros((self.width, self.height), dtype=bool)
        self.organisms: list[Organism] = self.build_initial_config()

    # ------------------------------------------------------------------
    # Setting up
    # ------------------------------------------------------------------

    def build_initial_config(self) -> list[Organism]:
        """Create the founding population, each with a fully random genome."""
        organisms = [Organism(self.config) for _ in range(self.n_organisms)]
        self.reset_grid(organisms)
        return organisms

    def reset_grid(self, organisms: Iterable[Organism]) -> None:
        """Clear the grid and scatter the given organisms over empty cells."""
        self.occupancy[:, :] = False
        self.step = 0
        for org in organisms:
            org.x = org.y = -1
            org.last_dx = org.last_dy = 0
            org.age = 0
            org.brain.reset()
            self.place(org, *self.random_empty_cell())

    def random_empty_cell(self) -> tuple[int, int]:
        """Pick an unoccupied cell, falling back to a scan if the grid is crowded."""
        for _ in range(100):
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            if not self.occupancy[x, y]:
                return x, y

        empty = np.argwhere(~self.occupancy)
        if len(empty) == 0:
            raise RuntimeError("no empty cells left: population exceeds grid size")
        x, y = empty[random.randrange(len(empty))]
        return int(x), int(y)

    # ------------------------------------------------------------------
    # Grid queries, used by the sensors
    # ------------------------------------------------------------------

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_occupied(self, x: int, y: int) -> bool:
        return bool(self.occupancy[x, y])

    def can_move_to(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and not self.occupancy[x, y]

    def population_density(self, x: int, y: int, radius: int) -> float:
        """
        Fraction of the cells around (x, y) that hold another organism.

        The neighbourhood is clipped at the walls, so an organism in a corner
        does not read as lonely just because part of its window is off-grid.
        """
        window = self.occupancy[
            max(0, x - radius) : min(self.width, x + radius + 1),
            max(0, y - radius) : min(self.height, y + radius + 1),
        ]
        neighbours = int(window.sum()) - 1  # discount the organism itself
        cells = window.size - 1
        return neighbours / cells if cells > 0 else 0.0

    def place(self, org: Organism, x: int, y: int) -> None:
        org.x, org.y = x, y
        self.occupancy[x, y] = True

    def relocate(self, org: Organism, x: int, y: int) -> None:
        self.occupancy[org.x, org.y] = False
        self.occupancy[x, y] = True
        org.x, org.y = x, y

    # ------------------------------------------------------------------
    # Running time
    # ------------------------------------------------------------------

    def update_organisms(self) -> None:
        """
        Advance every organism by one timestep.

        The order is shuffled each step so that no organism gets a permanent
        advantage in claiming contested cells.
        """
        order = list(self.organisms)
        random.shuffle(order)
        for org in order:
            org.act(self)
        self.step += 1

    def run_generation(self, on_step: Callable[[World, int], None] | None = None) -> int:
        """
        Run one full generation, then select and repopulate.

        `on_step` is called with (world, step) after each timestep, which is how
        `execute.py` draws the animation without the world knowing about it.

        Returns the number of organisms that met the survival criterion.
        """
        for step in range(self.config.steps_per_generation):
            self.update_organisms()
            if on_step is not None:
                on_step(self, step)

        survivors = self.select_survivors()
        self.reproduce_organisms(survivors)
        self.generation += 1
        return len(survivors)

    def select_survivors(self) -> list[Organism]:
        """The organisms that satisfy the survival criterion. Everyone else dies."""
        return [org for org in self.organisms if self.criterion(org.x, org.y, self)]

    def reproduce_organisms(self, survivors: list[Organism]) -> None:
        """
        Refill the population from the survivors and reset the grid.

        Parents are drawn at random with replacement, so a lone survivor can
        found an entire generation. With no survivors at all the run would end,
        so the world reseeds itself with fresh random genomes instead.
        """
        if survivors:
            children = [random.choice(survivors).reproduce() for _ in range(self.n_organisms)]
        else:
            children = [Organism(self.config) for _ in range(self.n_organisms)]

        self.organisms = children
        self.reset_grid(children)

    def positions(self) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Current x and y arrays for the whole population, for plotting."""
        count = len(self.organisms)
        xs = np.fromiter((org.x for org in self.organisms), dtype=int, count=count)
        ys = np.fromiter((org.y for org in self.organisms), dtype=int, count=count)
        return xs, ys

    def survival_zone_mask(self) -> npt.NDArray[np.bool_]:
        """
        A boolean grid of the cells that count as survivable.

        Rather than hard-coding a shape per criterion, this asks the criterion
        itself about every cell, so any new criterion draws itself correctly.
        """
        return np.array(
            [[self.criterion(x, y, self) for y in range(self.height)] for x in range(self.width)],
            dtype=bool,
        )


# ----------------------------------------------------------------------------
# Survival criteria
# ----------------------------------------------------------------------------
#
# A criterion takes a position and the world, and returns True if an organism
# ending the generation there gets to reproduce. This is the entire selection
# pressure -- swapping the criterion is the main dial for changing what evolves.


def left_edge(x: int, y: int, world: World) -> bool:
    """Survive by ending the generation in a band along the west wall."""
    return x < world.width * world.config.survival_zone_fraction


def right_edge(x: int, y: int, world: World) -> bool:
    """Survive by ending the generation in a band along the east wall."""
    return x >= world.width * (1 - world.config.survival_zone_fraction)


def centre(x: int, y: int, world: World) -> bool:
    """Survive by ending the generation inside a circle at the middle of the world."""
    radius = world.width * world.config.survival_zone_fraction
    dx = x - world.width / 2
    dy = y - world.height / 2
    return dx * dx + dy * dy <= radius * radius


def corners(x: int, y: int, world: World) -> bool:
    """Survive by ending the generation in any of the four corners."""
    span = world.width * world.config.survival_zone_fraction
    near_x = x < span or x >= world.width - span
    near_y = y < span or y >= world.height - span
    return near_x and near_y


CRITERIA: Final[dict[str, Criterion]] = {
    "left": left_edge,
    "right": right_edge,
    "centre": centre,
    "corners": corners,
}

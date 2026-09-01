"""
The organisms and the world they live in.

A generation runs for `Settings.steps_per_generation` timesteps. At the end of
it an objective decides who reproduces; everybody else dies. Survivors are
cloned with mutation until the population is full again, and the next
generation starts from scratch on a cleared grid.

The world holds three layers over the same grid:

- `barriers`    solid cells, fixed for the whole run
- `occupancy`   which cells hold a living organism, changing every step
- `pheromone`   scent, deposited by organisms and decaying every step

Barriers make navigation a real problem; pheromone is the only way one
organism can change what another perceives.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Final

import numpy as np

import barriers as barrier_layouts
import brain_utils
import reproduction
import settings
from capability_utils import Action, Sensor, read_sensors
from objectives import OBJECTIVES, Objective

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

    import numpy.typing as npt

    from brain_utils import Genome
    from settings import Settings

# The eight ways a step can point, used when an organism moves at random.
DIRECTIONS: Final = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)  # fmt: skip


class Organism:
    """
    One creature: a position on the grid, a genome, and the brain it builds.

    An organism does not know how to find anything out for itself -- every
    sense it has comes from `capability_utils`, and which of those senses it
    actually consults is decided by its genome.
    """

    __slots__ = (
        "age",
        "alive",
        "brain",
        "config",
        "energy",
        "genome",
        "last_dx",
        "last_dy",
        "lineage",
        "name",
        "progress",
        "upkeep",
        "x",
        "y",
    )

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
        self.alive = True
        self.energy = config.initial_energy

        self.lineage: int = -1
        """Which founding organism this one descends from. Set by the world."""

        # What this brain costs to run, every timestep, for as long as it lives.
        # Charged per *distinct sense consulted* rather than per gene, so wiring
        # the same sense twice is free but reaching for a new one is not.
        self.upkeep = config.metabolism + config.sense_cost * len(self.brain.needed_sensors)

        self.progress: dict[str, float] = {}
        """Scratch space for objectives that accumulate history over a generation."""

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

        # A quarter turn to the left maps (dx, dy) -> (-dy, dx).
        leftward = levels[Action.MOVE_LEFT]
        urge_x += leftward * -self.last_dy
        urge_y += leftward * self.last_dx

        if wander := levels[Action.MOVE_RANDOM]:
            dx, dy = random.choice(DIRECTIONS)
            urge_x += wander * dx
            urge_y += wander * dy

        # A positive 'stay' level damps whatever the other neurons wanted.
        stillness = max(0.0, levels[Action.STAY])
        urge_x *= 1.0 - stillness
        urge_y *= 1.0 - stillness

        spent = self.upkeep
        if (emission := levels[Action.EMIT_PHEROMONE]) > 0:
            world.deposit_pheromone(self.x, self.y, emission * self.config.pheromone_deposit)
            spent += emission * self.config.emit_cost

        before = (self.x, self.y)
        self.move(_urge_to_step(urge_x), _urge_to_step(urge_y), world)
        if (self.x, self.y) != before:
            spent += self.config.move_cost

        self.age += 1

        if self.config.energy_enabled:
            self.energy -= spent
            if self.energy <= 0.0:
                world.kill(self)

    def move(self, dx: int, dy: int, world: World) -> None:
        """
        Try to step by (dx, dy), refusing moves into walls, barriers or organisms.

        A blocked diagonal falls back to whichever single axis is still open,
        which lets organisms slide along an obstacle instead of sticking to it.
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

        # Fully boxed in: keep the heading so the blocked_* senses can report it.
        self.last_dx, self.last_dy = dx, dy

    # ------------------------------------------------------------------
    # Reproduction
    # ------------------------------------------------------------------

    def reproduce(self) -> Organism:
        """
        Produce one offspring by cloning this organism's genome with mutations.

        Kept for the asexual case and for direct use; `reproduction.py` owns
        the general question of who breeds with whom. The child is unplaced --
        the world decides where it starts.
        """
        child = Organism(self.config, genome=brain_utils.mutate(self.genome, self.config))
        child.lineage = self.lineage
        return child

    def __repr__(self) -> str:
        state = "" if self.alive else ", dead"
        return f"<Organism at ({self.x}, {self.y}){state}>"


def _clamp(value: float) -> float:
    """Hold a sensor reading inside [-1, 1]."""
    return max(-1.0, min(1.0, value))


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
    what we iterate over, the array is what makes "is that cell taken?", "how
    crowded is it here?" and "which way are the others?" cheap enough to ask on
    every timestep.
    """

    def __init__(
        self,
        config: Settings | None = None,
        objective: Objective | str | None = None,
        n_organisms: int | None = None,
        strategy: reproduction.Strategy | str | None = None,
    ) -> None:
        self.config = config or settings.DEFAULT
        self.objective = _resolve_objective(objective)
        self.strategy = reproduction.resolve(strategy or self.config.reproduction)
        self.n_organisms = n_organisms if n_organisms is not None else self.config.n_organisms

        self.extinct = False
        """Set when a generation leaves nobody able to breed. The run is over."""

        self.founding_lineages = 0
        """How many separate lines the run started with. Set by `found_lineages`."""

        self.width = self.config.width
        self.height = self.config.height

        self.generation = 0
        self.step = 0

        self.barriers: npt.NDArray[np.bool_] = barrier_layouts.build(
            self.config.barrier_layout, self.width, self.height
        )
        self.occupancy: npt.NDArray[np.bool_] = np.zeros((self.width, self.height), dtype=bool)
        self.pheromone: npt.NDArray[np.float32] = np.zeros(
            (self.width, self.height), dtype=np.float32
        )

        self.organisms: list[Organism] = self.build_initial_config()

    # ------------------------------------------------------------------
    # Setting up
    # ------------------------------------------------------------------

    def build_initial_config(self) -> list[Organism]:
        """Create the founding population, each with a fully random genome."""
        organisms = [Organism(self.config) for _ in range(self.n_organisms)]
        self.found_lineages(organisms)
        self.reset_grid(organisms)
        return organisms

    def found_lineages(self, organisms: list[Organism]) -> None:
        """
        Give each organism its own line, and remember how many there were.

        Keeping the founding count is what makes the survivor count mean
        anything: six lineages left is unremarkable out of ten and a near-total
        collapse out of four hundred.
        """
        for line, org in enumerate(organisms):
            org.lineage = line
        self.founding_lineages = len(organisms)

    @property
    def population(self) -> int:
        """How many organisms there currently are, living or not."""
        return len(self.organisms)

    @property
    def zone_fraction(self) -> float:
        """
        How big the survival zone is right now.

        Contracts by `zone_shrink_per_generation` each generation, so a
        solution that worked early stops working later, down to a floor that
        keeps the target from vanishing entirely.
        """
        base = self.config.survival_zone_fraction * self.objective.zone_scale
        shrink = (1.0 - self.config.zone_shrink_per_generation) ** self.generation
        return max(self.config.min_zone_fraction, base * shrink)

    def reset_grid(self, organisms: Iterable[Organism]) -> None:
        """Clear the grid and scatter the given organisms over empty cells."""
        self.occupancy[:, :] = False
        self.pheromone[:, :] = 0.0
        self.step = 0
        for org in organisms:
            org.x = org.y = -1
            org.last_dx = org.last_dy = 0
            org.age = 0
            org.alive = True
            org.energy = self.config.initial_energy
            org.progress.clear()
            org.brain.reset()
            self.place(org, *self.random_empty_cell())

    def random_empty_cell(self) -> tuple[int, int]:
        """Pick a free, non-solid cell, falling back to a scan if the grid is crowded."""
        for _ in range(100):
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            if not self.occupancy[x, y] and not self.barriers[x, y]:
                return x, y

        free = np.argwhere(~(self.occupancy | self.barriers))
        if len(free) == 0:
            raise RuntimeError("no free cells left: population exceeds the open grid")
        x, y = free[random.randrange(len(free))]
        return int(x), int(y)

    # ------------------------------------------------------------------
    # Grid queries, used by the sensors
    # ------------------------------------------------------------------

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_occupied(self, x: int, y: int) -> bool:
        return bool(self.occupancy[x, y])

    def can_move_to(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and not self.occupancy[x, y] and not self.barriers[x, y]

    def _window_bounds(self, x: int, y: int, radius: int) -> tuple[int, int, int, int]:
        """The neighbourhood around a cell, clipped to the grid."""
        return (
            max(0, x - radius),
            min(self.width, x + radius + 1),
            max(0, y - radius),
            min(self.height, y + radius + 1),
        )

    def population_density(self, x: int, y: int, radius: int) -> float:
        """
        Fraction of the cells around (x, y) that hold another organism.

        The neighbourhood is clipped at the walls, so an organism in a corner
        does not read as lonely just because part of its window is off-grid.
        """
        x0, x1, y0, y1 = self._window_bounds(x, y, radius)
        window = self.occupancy[x0:x1, y0:y1]
        neighbours = int(window.sum()) - 1  # discount the organism itself
        cells = window.size - 1
        return neighbours / cells if cells > 0 else 0.0

    def neighbour_gradient(self, x: int, y: int, radius: int) -> tuple[float, float]:
        """
        Which way the neighbours lie, as a signed pair in [-1, 1].

        (+1, 0) means every neighbour in range is east; (0, 0) means they are
        balanced, or that there are none. This is the directional information
        `population_density` throws away.
        """
        x0, x1, y0, y1 = self._window_bounds(x, y, radius)
        window = self.occupancy[x0:x1, y0:y1]
        total = int(window.sum()) - 1
        if total <= 0:
            return 0.0, 0.0

        east = int(self.occupancy[x + 1 : x1, y0:y1].sum())
        west = int(self.occupancy[x0:x, y0:y1].sum())
        north = int(self.occupancy[x0:x1, y + 1 : y1].sum())
        south = int(self.occupancy[x0:x1, y0:y].sum())
        return _clamp((east - west) / total), _clamp((north - south) / total)

    def nearest_neighbour(self, x: int, y: int, radius: int) -> float:
        """
        Closeness of the nearest other organism: 1 when adjacent, 0 when none is in range.

        The empty case is the common one in a sparse world, so it is settled
        with a single array sum before doing any real work.
        """
        x0, x1, y0, y1 = self._window_bounds(x, y, radius)
        window = self.occupancy[x0:x1, y0:y1]
        if int(window.sum()) <= 1:
            return 0.0

        others = np.argwhere(window)
        dx = others[:, 0] + x0 - x
        dy = others[:, 1] + y0 - y
        distances = np.maximum(np.abs(dx), np.abs(dy))  # Chebyshev: one step per ring
        nearest = int(distances[distances > 0].min())
        return max(0.0, 1.0 - (nearest - 1) / radius)

    # ------------------------------------------------------------------
    # The pheromone layer
    # ------------------------------------------------------------------

    def pheromone_at(self, x: int, y: int) -> float:
        """Scent on one cell, saturating at 1."""
        return min(float(self.pheromone[x, y]), 1.0)

    def pheromone_gradient(self, x: int, y: int, radius: int) -> tuple[float, float]:
        """Which way the scent gets stronger, as a signed pair in [-1, 1]."""
        x0, x1, y0, y1 = self._window_bounds(x, y, radius)
        total = float(self.pheromone[x0:x1, y0:y1].sum())
        if total <= 0.0:
            return 0.0, 0.0

        east = float(self.pheromone[x + 1 : x1, y0:y1].sum())
        west = float(self.pheromone[x0:x, y0:y1].sum())
        north = float(self.pheromone[x0:x1, y + 1 : y1].sum())
        south = float(self.pheromone[x0:x1, y0:y].sum())
        # The sub-sums are float32 while the total is not, so rounding can push
        # a ratio a hair past 1. Clamp, or a sensor quietly breaks its contract.
        return _clamp((east - west) / total), _clamp((north - south) / total)

    def deposit_pheromone(self, x: int, y: int, amount: float) -> None:
        """Lay scent on a cell. Deposits accumulate; sensing saturates at 1."""
        self.pheromone[x, y] += amount

    # ------------------------------------------------------------------
    # Placing and removing organisms
    # ------------------------------------------------------------------

    def place(self, org: Organism, x: int, y: int) -> None:
        org.x, org.y = x, y
        self.occupancy[x, y] = True

    def relocate(self, org: Organism, x: int, y: int) -> None:
        self.occupancy[org.x, org.y] = False
        self.occupancy[x, y] = True
        org.x, org.y = x, y

    def kill(self, org: Organism) -> None:
        """Remove an organism from play. It keeps its position but frees its cell."""
        if not org.alive:
            return
        org.alive = False
        self.occupancy[org.x, org.y] = False

    def living(self) -> list[Organism]:
        return [org for org in self.organisms if org.alive]

    # ------------------------------------------------------------------
    # Running time
    # ------------------------------------------------------------------

    def update_organisms(self) -> None:
        """
        Advance every living organism by one timestep, then age the world.

        The order is shuffled each step so that no organism gets a permanent
        advantage in claiming contested cells.
        """
        order = self.living()
        random.shuffle(order)
        for org in order:
            org.act(self)

        self.pheromone *= self.config.pheromone_decay
        self.step += 1

    def iter_generation(self) -> Generator[int, None, int]:
        """
        Run one generation as a generator, pausing after each timestep.

        Yields the step number, and returns the number of survivors once
        exhausted. Pausing between steps is what lets a caller that cannot
        block -- the web server, streaming frames -- drive the simulation at
        its own pace without the world knowing anything about it.

        Selection and repopulation happen when the generator finishes, so an
        abandoned generator leaves the world mid-generation and unchanged.
        """
        if self.extinct:
            return 0

        self.objective.begin_generation(self)

        for step in range(self.config.steps_per_generation):
            self.update_organisms()
            self.objective.advance(self)
            for org in self.organisms:
                if org.alive:
                    self.objective.observe(org, self)
            yield step

        survivors = self.select_survivors()
        self.reproduce_organisms(survivors)
        self.generation += 1
        return len(survivors)

    def run_generation(self, on_step: Callable[[World, int], None] | None = None) -> int:
        """
        Run one full generation, then select and repopulate.

        `on_step` is called with (world, step) after each timestep, which is how
        `execute.py` draws the animation without the world knowing about it.

        Returns the number of organisms that met the objective.
        """
        generation = self.iter_generation()
        while True:
            try:
                step = next(generation)
            except StopIteration as finished:
                return finished.value
            if on_step is not None:
                on_step(self, step)

    def select_survivors(self) -> list[Organism]:
        """The organisms that satisfy the objective. Everyone else dies."""
        return [org for org in self.organisms if org.alive and self.objective.survives(org, self)]

    def reproduce_organisms(self, survivors: list[Organism]) -> None:
        """
        Build the next generation from the survivors and reset the grid.

        The population is earned rather than refilled: it grows when the
        generation went well and shrinks when it did not. If nobody survives --
        or, under sexual reproduction, nobody found a partner -- the population
        is gone and the run is over.
        """
        children = reproduction.next_generation(
            survivors, self.strategy, self.config, population=self.population
        )
        if not children:
            self.extinct = True
            self.organisms = []
            self.occupancy[:, :] = False
            return

        # More offspring than open cells would loop forever looking for room.
        room = int((~self.barriers).sum())
        if len(children) > room:
            children = children[:room]

        self.organisms = children
        self.reset_grid(children)

    def positions(self) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Current x and y arrays for the living population, for plotting."""
        alive = self.living()
        xs = np.fromiter((org.x for org in alive), dtype=int, count=len(alive))
        ys = np.fromiter((org.y for org in alive), dtype=int, count=len(alive))
        return xs, ys


def _resolve_objective(objective: Objective | str | None) -> Objective:
    """Accept an objective, the name of one, or nothing at all."""
    if objective is None:
        return OBJECTIVES["left"]
    if isinstance(objective, str):
        try:
            return OBJECTIVES[objective]
        except KeyError:
            known = ", ".join(OBJECTIVES)
            raise ValueError(f"unknown objective {objective!r}; try one of: {known}") from None
    return objective

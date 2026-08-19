"""
Sanity checks for the simulation. Run with:

    uv run pytest

These are deliberately about invariants rather than exact values -- the whole
system is stochastic, so anything asserting a specific outcome would be flaky.
What must always hold is that organisms stay on the grid, never share a cell,
and that mutation only ever produces genes that can be wired up.
"""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest

import brain_utils
from brain_utils import Brain, Gene, Sink, Source
from capability_utils import Action, Sensor
from organism import CRITERIA, Organism, World
from settings import Settings


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    """Seed both generators so a failure can be reproduced."""
    random.seed(0)
    np.random.seed(0)


@pytest.fixture
def config() -> Settings:
    """A small, fast world for tests that do not care about scale."""
    return replace(Settings(), n_organisms=40, steps_per_generation=30)


# ----------------------------------------------------------------------------
# The world and its occupancy grid
# ----------------------------------------------------------------------------


def test_organisms_stay_in_bounds(config: Settings) -> None:
    """Organisms must never walk off the grid, however long the run."""
    world = World(config=config)
    for _ in range(300):
        world.update_organisms()
        for org in world.organisms:
            assert world.in_bounds(org.x, org.y), f"organism escaped to ({org.x}, {org.y})"


def test_organisms_do_not_get_stuck_at_the_walls(config: Settings) -> None:
    """
    An organism that reaches a wall must still be able to move along it.

    This is the failure the original `move` had: its bounds check froze any
    organism that stepped onto the boundary.
    """
    world = World(config=config, n_organisms=1)
    org = world.organisms[0]
    middle = world.height // 2
    world.relocate(org, 0, middle)

    org.move(-1, 0, world)
    assert org.x == 0, "organism should be held at the wall, not pushed through"

    org.move(0, 1, world)
    assert org.y == middle + 1, "organism at a wall should still move along it"


def test_one_organism_per_cell(config: Settings) -> None:
    """Occupancy must stay exclusive, and the grid must match the population."""
    world = World(config=config)
    for _ in range(120):
        world.update_organisms()

    cells = {(org.x, org.y) for org in world.organisms}
    assert len(cells) == len(world.organisms), "two organisms share a cell"
    assert int(world.occupancy.sum()) == len(world.organisms), (
        "occupancy grid disagrees with the population"
    )


def test_blocked_moves_are_refused(config: Settings) -> None:
    """An organism cannot step onto a cell another organism already holds."""
    world = World(config=config, n_organisms=2)
    first, second = world.organisms
    world.relocate(first, 10, 10)
    world.relocate(second, 11, 10)

    first.move(1, 0, world)
    assert (first.x, first.y) == (10, 10), "organism moved into an occupied cell"


def test_population_density_ignores_the_organism_itself(config: Settings) -> None:
    """An organism alone in an empty world should read zero density."""
    world = World(config=config, n_organisms=1)
    world.relocate(world.organisms[0], 50, 50)
    assert world.population_density(50, 50, 4) == 0.0, "an organism sensed itself"


def test_population_density_saturates_when_crowded(config: Settings) -> None:
    """A fully occupied neighbourhood should read as fully dense."""
    world = World(config=config, n_organisms=1)
    world.occupancy[48:53, 48:53] = True
    assert world.population_density(50, 50, 2) == 1.0


def test_every_organism_is_placed_on_the_grid(config: Settings) -> None:
    """No organism may be left without a cell after the world is built."""
    world = World(config=config)
    assert all(org.placed for org in world.organisms)


# ----------------------------------------------------------------------------
# Genomes and brains
# ----------------------------------------------------------------------------


def test_mutation_only_produces_valid_genes(config: Settings) -> None:
    """
    Every gene must point at endpoints that exist.

    Mutation can flip an endpoint's *kind*, and each kind has a different id
    range, so this is the property most likely to break when the Sensor or
    Action enums change.
    """
    genome = brain_utils.random_genome(config)
    for _ in range(400):
        genome = brain_utils.mutate(genome, config, rate=0.5)
        for gene in genome:
            if gene.source_kind is Source.SENSOR:
                assert gene.source_id in set(Sensor)
            else:
                assert 0 <= gene.source_id < config.n_inner_neurons

            if gene.sink_kind is Sink.INNER:
                assert 0 <= gene.sink_id < config.n_inner_neurons
            else:
                assert gene.sink_id in set(Action)

            assert abs(gene.weight) <= config.max_weight


def test_genes_are_immutable(config: Settings) -> None:
    """Frozen genes are what make sharing them between parent and child safe."""
    gene = brain_utils.random_gene(config)
    with pytest.raises(AttributeError):
        gene.weight = 0.0  # type: ignore[misc]


def test_mutation_does_not_touch_the_parent(config: Settings) -> None:
    """Reproduction must derive a new genome, never edit the parent's."""
    parent = brain_utils.random_genome(config)
    snapshot = list(parent)
    brain_utils.mutate(parent, config, rate=1.0)
    assert list(parent) == snapshot, "mutation modified the parent genome"


def test_brains_produce_one_bounded_level_per_action(config: Settings) -> None:
    """`think` must return a usable level for every action, every time."""
    world = World(config=config)
    for _ in range(20):
        for org in world.organisms:
            levels = org.brain.think(org, world)
            assert len(levels) == len(Action)
            assert all(-1.0 <= level <= 1.0 for level in levels)
        world.update_organisms()


def test_brains_only_read_the_sensors_they_are_wired_to(config: Settings) -> None:
    """The lazy sensor set must match what the genome actually references."""
    genome = brain_utils.random_genome(config)
    brain = Brain(genome, config)
    expected = {Sensor(g.source_id) for g in genome if g.source_kind is Source.SENSOR}
    assert set(brain.needed_sensors) == expected


def test_inner_neurons_carry_state_between_timesteps(config: Settings) -> None:
    """
    A recurrent loop with no sensor input must still drive an action.

    This is what makes memory reachable by evolution, so it is worth pinning
    rather than trusting to a random genome to exercise it.
    """
    genome = (
        Gene(Source.SENSOR, Sensor.BIAS, Sink.INNER, 0, 2.0),
        Gene(Source.INNER, 0, Sink.INNER, 1, 2.0),
        Gene(Source.INNER, 1, Sink.ACTION, Action.MOVE_X, 2.0),
    )
    world = World(config=config, n_organisms=1)
    org = Organism(config, genome=genome)
    world.place(org, 5, 5)

    # Inner neuron 1 starts at zero, so the action is silent on the first step
    # and only responds once the signal has propagated through the loop.
    assert world  # keep the world alive for the sensor calls
    first = org.brain.think(org, world)[Action.MOVE_X]
    second = org.brain.think(org, world)[Action.MOVE_X]
    assert first == pytest.approx(0.0)
    assert second > 0.5


def test_perceive_reports_every_sensor(config: Settings) -> None:
    """The debugging view must cover the full sensor list, in range."""
    world = World(config=config)
    knowledge = world.organisms[0].perceive(world)
    assert set(knowledge) == set(Sensor)
    for sensor, value in knowledge.items():
        assert -1.0 <= value <= 1.0, f"{sensor} out of range: {value}"


def test_gene_describe_names_its_endpoints(config: Settings) -> None:
    """The wiring readout is the main way behaviour gets explained, so pin its shape."""
    gene = Gene(Source.SENSOR, Sensor.BORDER_DISTANCE, Sink.ACTION, Action.MOVE_X, -2.134)
    assert gene.describe() == "border_distance --(-2.13)--> move_x"

    recurrent = Gene(Source.INNER, 2, Sink.INNER, 3, 1.5)
    assert recurrent.describe() == "inner_2 --(+1.50)--> inner_3"


# ----------------------------------------------------------------------------
# Selection and reproduction
# ----------------------------------------------------------------------------


def test_a_generation_preserves_population_size(config: Settings) -> None:
    """However selection goes, the next generation must be full and placed."""
    world = World(config=config)
    for _ in range(3):
        world.run_generation()
        assert len(world.organisms) == config.n_organisms
        for org in world.organisms:
            assert org.placed, "organism was never placed"
            assert org.age == 0, "a new generation should start at age zero"


def test_extinction_reseeds_rather_than_ending_the_run(config: Settings) -> None:
    """With no survivors the world must refill itself instead of emptying."""
    world = World(config=config, criterion=lambda x, y, w: False)
    world.run_generation()
    assert len(world.organisms) == config.n_organisms


@pytest.mark.parametrize("name", sorted(CRITERIA))
def test_survivors_are_exactly_those_in_the_zone(name: str, config: Settings) -> None:
    """Selection must return every organism inside the survival zone, and no others."""
    criterion = CRITERIA[name]
    world = World(config=config, criterion=criterion)
    for _ in range(30):
        world.update_organisms()

    survivors = set(map(id, world.select_survivors()))
    for org in world.organisms:
        assert criterion(org.x, org.y, world) == (id(org) in survivors)


@pytest.mark.parametrize("name", sorted(CRITERIA))
def test_survival_zone_mask_agrees_with_the_criterion(name: str, config: Settings) -> None:
    """The shading the animation draws must match the selection actually applied."""
    world = World(config=config, criterion=CRITERIA[name])
    mask = world.survival_zone_mask()
    assert mask.shape == (world.width, world.height)
    assert mask.any(), f"{name}: survival zone is empty"
    for x, y in [(0, 0), (world.width - 1, world.height - 1), (world.width // 2, 3)]:
        assert bool(mask[x, y]) == CRITERIA[name](x, y, world)


def test_a_child_inherits_its_parents_wiring(config: Settings) -> None:
    """At a zero mutation rate, offspring must be exact copies."""
    parent = Organism(config)
    child = Organism(config, genome=brain_utils.mutate(parent.genome, config, rate=0.0))
    assert child.genome == parent.genome
    assert child.brain.describe() == parent.brain.describe()


def test_evolution_improves_survival() -> None:
    """
    The whole point: survival must climb well above where random genomes start.

    Slow-ish, but a simulation that no longer evolves is the one failure that
    every other test here would happily pass through.
    """
    config = replace(Settings(), n_organisms=150, steps_per_generation=100)
    world = World(config=config, criterion=CRITERIA["left"])

    first = world.run_generation()
    for _ in range(19):
        last = world.run_generation()

    assert last > first, f"no improvement: started at {first}, ended at {last}"
    assert last / config.n_organisms > 0.6, f"only {last}/{config.n_organisms} survived"

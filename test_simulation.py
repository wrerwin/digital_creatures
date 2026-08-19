"""
Sanity checks for the simulation. Run with:

    uv run pytest

These are deliberately about invariants rather than exact values -- the whole
system is stochastic, so anything asserting a specific outcome would be flaky.
What must always hold is that organisms stay on the grid, never share a cell or
stand inside a wall, that mutation only produces genes that can be wired up,
and that every objective can actually be satisfied.
"""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import barriers
import brain_utils
import inspect_utils
import objectives
from brain_utils import Brain, Gene, Sink, Source
from capability_utils import SENSOR_FUNCTIONS, Action, Sensor
from objectives import OBJECTIVES, Hazard, StayInZone, VisitInOrder
from organism import Organism, World
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


def lone_world(config: Settings, **kwargs: object) -> tuple[World, Organism]:
    """A world holding exactly one organism, parked in the middle."""
    world = World(config=config, n_organisms=1, **kwargs)  # type: ignore[arg-type]
    org = world.organisms[0]
    world.relocate(org, world.width // 2, world.height // 2)
    return world, org


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
    world, org = lone_world(config)
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

    cells = {(org.x, org.y) for org in world.living()}
    assert len(cells) == len(world.living()), "two organisms share a cell"
    assert int(world.occupancy.sum()) == len(world.living()), (
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


def test_every_organism_is_placed_on_the_grid(config: Settings) -> None:
    """No organism may be left without a cell after the world is built."""
    world = World(config=config)
    assert all(org.placed for org in world.organisms)


# ----------------------------------------------------------------------------
# Barriers
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("layout", sorted(barriers.LAYOUTS))
def test_layouts_have_the_right_shape_and_leave_room(layout: str) -> None:
    """Every layout must fit the world and leave most of it walkable."""
    grid = barriers.build(layout, 100, 100)
    assert grid.shape == (100, 100)
    assert grid.dtype == np.bool_
    assert grid.mean() < 0.5, f"{layout} fills too much of the world"


def test_unknown_layout_is_reported_clearly() -> None:
    with pytest.raises(ValueError, match="unknown barrier layout"):
        barriers.build("swiss-cheese", 20, 20)


@pytest.mark.parametrize("layout", sorted(barriers.LAYOUTS))
def test_organisms_never_stand_inside_a_barrier(layout: str, config: Settings) -> None:
    """Placement and movement must both respect solid cells."""
    world = World(config=replace(config, barrier_layout=layout))
    for _ in range(60):
        world.update_organisms()
        for org in world.living():
            assert not world.barriers[org.x, org.y], f"{layout}: organism inside a wall"


def test_barriers_block_movement(config: Settings) -> None:
    """A solid cell refuses entry just as a world edge does."""
    world, org = lone_world(config)
    world.barriers[org.x + 1, org.y] = True
    assert not world.can_move_to(org.x + 1, org.y)

    before = org.x
    org.move(1, 0, world)
    assert org.x == before, "organism walked into a barrier"


# ----------------------------------------------------------------------------
# Senses
# ----------------------------------------------------------------------------


def test_every_sensor_stays_in_range(config: Settings) -> None:
    """A sensor outside [-1, 1] would quietly dominate every brain that used it."""
    world = World(config=replace(config, barrier_layout="pillars"))
    for _ in range(25):
        world.update_organisms()
        for org in world.living():
            for sensor, value in org.perceive(world).items():
                assert -1.0 <= value <= 1.0, f"{sensor} out of range: {value}"


def test_perceive_reports_every_sensor(config: Settings) -> None:
    """The debugging view must cover the full sensor list."""
    world = World(config=config)
    assert set(world.organisms[0].perceive(world)) == set(Sensor)


def test_neighbour_gradient_points_at_the_neighbours(config: Settings) -> None:
    """The directional sense must actually be directional."""
    world, org = lone_world(config)
    x, y = org.x, org.y

    assert world.neighbour_gradient(x, y, 6) == (0.0, 0.0), "alone should read as balanced"

    world.occupancy[x + 3, y] = True
    east, north = world.neighbour_gradient(x, y, 6)
    assert east == pytest.approx(1.0)
    assert north == pytest.approx(0.0)

    world.occupancy[x - 3, y] = True
    east, _ = world.neighbour_gradient(x, y, 6)
    assert east == pytest.approx(0.0), "neighbours on both sides should cancel"

    world.occupancy[x, y + 2] = True
    _, north = world.neighbour_gradient(x, y, 6)
    assert north > 0


def test_nearest_neighbour_falls_off_with_distance(config: Settings) -> None:
    """1 when adjacent, 0 when nothing is in range, decreasing in between."""
    world, org = lone_world(config)
    x, y = org.x, org.y
    assert world.nearest_neighbour(x, y, 6) == 0.0

    world.occupancy[x + 1, y] = True
    assert world.nearest_neighbour(x, y, 6) == pytest.approx(1.0)

    world.occupancy[x + 1, y] = False
    world.occupancy[x + 4, y] = True
    close = world.nearest_neighbour(x, y, 6)
    assert 0.0 < close < 1.0


def test_population_density_ignores_the_organism_itself(config: Settings) -> None:
    """An organism alone in an empty world should read zero density."""
    world, org = lone_world(config)
    assert world.population_density(org.x, org.y, 4) == 0.0, "an organism sensed itself"


def test_population_density_saturates_when_crowded(config: Settings) -> None:
    """A fully occupied neighbourhood should read as fully dense."""
    world, org = lone_world(config)
    x, y = org.x, org.y
    world.occupancy[x - 2 : x + 3, y - 2 : y + 3] = True
    assert world.population_density(x, y, 2) == 1.0


def test_blocked_senses_rotate_with_the_heading(config: Settings) -> None:
    """
    Left and right must be relative to where the organism is going.

    Heading east, 'left' is north -- getting the rotation backwards would give
    evolution a consistently mirrored world, which is hard to spot by eye.
    """
    world, org = lone_world(config)
    org.last_dx, org.last_dy = 1, 0  # heading east

    world.barriers[org.x, org.y + 1] = True  # a wall to the north
    assert SENSOR_FUNCTIONS[Sensor.BLOCKED_LEFT](org, world) == 1.0
    assert SENSOR_FUNCTIONS[Sensor.BLOCKED_RIGHT](org, world) == 0.0
    assert SENSOR_FUNCTIONS[Sensor.BLOCKED_FORWARD](org, world) == 0.0

    world.barriers[org.x, org.y + 1] = False
    world.barriers[org.x + 1, org.y] = True  # a wall dead ahead
    assert SENSOR_FUNCTIONS[Sensor.BLOCKED_FORWARD](org, world) == 1.0


def test_a_still_organism_reads_nothing_as_blocked(config: Settings) -> None:
    """With no heading there is no forward, left or right to report."""
    world, org = lone_world(config)
    org.last_dx = org.last_dy = 0
    for sensor in (Sensor.BLOCKED_FORWARD, Sensor.BLOCKED_LEFT, Sensor.BLOCKED_RIGHT):
        assert SENSOR_FUNCTIONS[sensor](org, world) == 0.0


# ----------------------------------------------------------------------------
# The pheromone layer
# ----------------------------------------------------------------------------


def test_pheromone_saturates_but_accumulates(config: Settings) -> None:
    """Sensing tops out at 1 even though deposits keep stacking."""
    world, org = lone_world(config)
    world.deposit_pheromone(org.x, org.y, 0.4)
    assert world.pheromone_at(org.x, org.y) == pytest.approx(0.4)

    for _ in range(10):
        world.deposit_pheromone(org.x, org.y, 0.4)
    assert world.pheromone_at(org.x, org.y) == 1.0


def test_pheromone_decays_over_time(config: Settings) -> None:
    """Trails must fade, or the whole grid saturates and the sense goes blind."""
    world, org = lone_world(config)
    world.deposit_pheromone(org.x, org.y, 1.0)
    before = world.pheromone_at(org.x, org.y)

    for _ in range(5):
        world.update_organisms()

    assert world.pheromone_at(org.x, org.y) < before


def test_pheromone_gradient_points_up_the_trail(config: Settings) -> None:
    """Following a scent is only possible if the gradient has the right sign."""
    world, org = lone_world(config)
    x, y = org.x, org.y
    assert world.pheromone_gradient(x, y, 6) == (0.0, 0.0), "unscented should read as balanced"

    world.deposit_pheromone(x + 3, y, 1.0)
    east, north = world.pheromone_gradient(x, y, 6)
    assert east == pytest.approx(1.0)
    assert north == pytest.approx(0.0)

    world.deposit_pheromone(x, y - 3, 1.0)
    _, north = world.pheromone_gradient(x, y, 6)
    assert north < 0


def test_emitting_leaves_scent_where_the_organism_stands(config: Settings) -> None:
    """The emit action has to reach the grid, not just the brain's output."""
    genome = (Gene(Source.SENSOR, Sensor.BIAS, Sink.ACTION, Action.EMIT_PHEROMONE, 4.0),)
    world, _ = lone_world(config)
    org = Organism(config, genome=genome)
    world.reset_grid([*world.organisms, org])

    x, y = org.x, org.y
    org.act(world)
    assert world.pheromone[x, y] > 0.0


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
        for org in world.living():
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
    rather than trusting a random genome to exercise it.
    """
    genome = (
        Gene(Source.SENSOR, Sensor.BIAS, Sink.INNER, 0, 2.0),
        Gene(Source.INNER, 0, Sink.INNER, 1, 2.0),
        Gene(Source.INNER, 1, Sink.ACTION, Action.MOVE_X, 2.0),
    )
    world, _ = lone_world(config)
    org = Organism(config, genome=genome)
    world.place(org, 5, 5)

    # Inner neuron 1 starts at zero, so the action is silent on the first step
    # and only responds once the signal has propagated through the loop.
    first = org.brain.think(org, world)[Action.MOVE_X]
    second = org.brain.think(org, world)[Action.MOVE_X]
    assert first == pytest.approx(0.0)
    assert second > 0.5


def test_gene_describe_names_its_endpoints() -> None:
    """The wiring readout is the main way behaviour gets explained, so pin its shape."""
    gene = Gene(Source.SENSOR, Sensor.BORDER_DISTANCE, Sink.ACTION, Action.MOVE_X, -2.134)
    assert gene.describe() == "border_distance --(-2.13)--> move_x"

    recurrent = Gene(Source.INNER, 2, Sink.INNER, 3, 1.5)
    assert recurrent.describe() == "inner_2 --(+1.50)--> inner_3"


# ----------------------------------------------------------------------------
# Objectives
# ----------------------------------------------------------------------------


def test_a_new_generation_starts_clean(config: Settings) -> None:
    """
    However selection goes, whoever survives into the next generation is ready.

    Population size is no longer fixed -- it is earned, and tested in
    `test_evolution.py` -- but everyone alive must be placed and reset.
    """
    world = World(config=config)
    for _ in range(3):
        world.run_generation()
        if world.extinct:
            break
        assert world.population <= config.carrying_capacity
        for org in world.organisms:
            assert org.placed, "organism was never placed"
            assert org.age == 0, "a new generation should start at age zero"
            assert org.alive, "a new generation should start alive"
            assert org.energy == config.initial_energy, "energy should be refilled"


def test_an_impossible_objective_ends_the_run(config: Settings) -> None:
    """
    With nobody able to breed, the population is gone and stays gone.

    This used to reseed from fresh random genomes, which quietly hid the fact
    that a run had failed. Extinction is now the honest outcome.
    """

    class Impossible(objectives.Objective):
        name = "impossible"

        def survives(self, org: Organism, world: World) -> bool:
            return False

    world = World(config=config, objective=Impossible())
    world.run_generation()

    assert world.extinct
    assert world.organisms == []


@pytest.mark.parametrize("name", list(OBJECTIVES))
def test_every_objective_runs_and_can_be_drawn(name: str, config: Settings) -> None:
    """Each objective must survive a generation and describe its own zones."""
    world = World(config=config, objective=name)
    survivors = world.run_generation()
    assert 0 <= survivors <= config.n_organisms

    for shading in world.objective.zones(world):
        assert shading.mask.shape == (world.width, world.height)
        assert shading.mask.any(), f"{name}: zone {shading.label} is empty"


def test_unknown_objective_is_reported_clearly(config: Settings) -> None:
    with pytest.raises(ValueError, match="unknown objective"):
        World(config=config, objective="become-immortal")


def test_reach_zone_selects_exactly_those_in_the_zone(config: Settings) -> None:
    """Selection must return every organism inside the zone, and no others."""
    objective = OBJECTIVES["left"]
    world = World(config=config, objective=objective)
    for _ in range(30):
        world.update_organisms()

    survivors = set(map(id, world.select_survivors()))
    for org in world.organisms:
        assert objective.survives(org, world) == (id(org) in survivors)


def test_stay_in_zone_needs_time_not_just_arrival(config: Settings) -> None:
    """Ending in the zone is not enough if the organism only just got there."""
    objective = StayInZone("stay-test", objectives.left_edge, fraction=0.5)
    world = World(config=config, objective=objective, n_organisms=1)
    org = world.organisms[0]

    objective.begin_generation(world)
    world.relocate(org, 1, 5)  # inside the zone, but with no history
    assert not objective.survives(org, world)

    required = int(0.5 * config.steps_per_generation)
    for _ in range(required):
        objective.observe(org, world)
    assert objective.survives(org, world)


def test_visit_in_order_requires_both_halves(config: Settings) -> None:
    """Finishing in the second zone without touching the first must not count."""
    objective = VisitInOrder("order-test", objectives.right_edge, objectives.left_edge)
    world = World(config=config, objective=objective, n_organisms=1)
    org = world.organisms[0]

    objective.begin_generation(world)
    world.relocate(org, 1, 5)  # in the finishing zone, never visited the first
    assert not objective.survives(org, world)

    world.relocate(org, world.width - 2, 5)  # touch the first zone
    objective.observe(org, world)
    assert not objective.survives(org, world), "visiting the first zone alone is not enough"

    world.relocate(org, 1, 5)
    assert objective.survives(org, world)


def test_hazard_kills_and_frees_the_cell(config: Settings) -> None:
    """A caught organism must stop acting and stop blocking the cell it held."""
    hazard = Hazard()
    world = World(config=config, objective=hazard, n_organisms=1)
    org = world.organisms[0]

    hazard.begin_generation(world)
    world.relocate(org, int(hazard.x), int(hazard.y))
    hazard.advance(world)

    assert not org.alive
    assert not world.is_occupied(org.x, org.y), "a dead organism still blocks its cell"
    assert org not in world.living()
    assert not hazard.survives(org, world)


def test_dead_organisms_stop_moving(config: Settings) -> None:
    """The update loop must skip the dead rather than quietly animating corpses."""
    world = World(config=config)
    victim = world.organisms[0]
    world.kill(victim)
    where = (victim.x, victim.y)

    for _ in range(20):
        world.update_organisms()

    assert (victim.x, victim.y) == where


def test_a_child_inherits_its_parents_wiring(config: Settings) -> None:
    """At a zero mutation rate, offspring must be exact copies."""
    parent = Organism(config)
    child = Organism(config, genome=brain_utils.mutate(parent.genome, config, rate=0.0))
    assert child.genome == parent.genome
    assert child.brain.describe() == parent.brain.describe()


# ----------------------------------------------------------------------------
# Capabilities: which senses and actions evolution is allowed to use
# ----------------------------------------------------------------------------


def test_disabled_capabilities_are_never_wired_up(config: Settings) -> None:
    """
    The whole point of the capability switches: a disabled sense must not appear.

    Checked across heavy mutation, because gene creation and gene mutation are
    separate code paths and only one of them failing would be easy to miss.
    """
    limited = config.with_capabilities(
        sensors=[Sensor.X_POSITION, Sensor.BIAS],
        actions=[Action.MOVE_X, Action.STAY],
    )

    genome = brain_utils.random_genome(limited)
    for _ in range(200):
        genome = brain_utils.mutate(genome, limited, rate=0.5)
        for gene in genome:
            if gene.source_kind is Source.SENSOR:
                assert Sensor(gene.source_id) in limited.enabled_sensors
            if gene.sink_kind is Sink.ACTION:
                assert Action(gene.sink_id) in limited.enabled_actions


def test_a_restricted_population_only_consults_what_it_has(config: Settings) -> None:
    """A whole world of restricted creatures must stay within its capabilities."""
    limited = replace(config, steps_per_generation=20).with_capabilities(
        sensors=[Sensor.Y_POSITION], actions=[Action.MOVE_Y]
    )

    world = World(config=limited)
    world.run_generation()
    for org in world.organisms:
        assert set(org.brain.needed_sensors) <= {Sensor.Y_POSITION}


def test_capabilities_cannot_be_emptied(config: Settings) -> None:
    """
    A creature with no senses or no actions cannot evolve at all.

    Rejected up front, because the alternative is an IndexError from deep
    inside gene creation that says nothing about the real mistake.
    """
    with pytest.raises(ValueError, match="at least one sensor"):
        config.with_capabilities(sensors=[])
    with pytest.raises(ValueError, match="at least one action"):
        config.with_capabilities(actions=[])


def test_capability_order_does_not_change_a_seeded_run(config: Settings) -> None:
    """
    The same selection ticked in a different order must give the same run.

    The UI hands back whatever order the checkboxes were in, so normalising is
    what keeps a shared seed reproducible.
    """
    forwards = config.with_capabilities(sensors=[Sensor.BIAS, Sensor.AGE])
    backwards = config.with_capabilities(sensors=[Sensor.AGE, Sensor.BIAS])
    assert forwards == backwards

    random.seed(11)
    first = brain_utils.random_genome(forwards)
    random.seed(11)
    second = brain_utils.random_genome(backwards)
    assert first == second


def test_restricting_capabilities_leaves_the_enums_alone(config: Settings) -> None:
    """Disabling a sense for one run must not affect any other world."""
    config.with_capabilities(sensors=[Sensor.BIAS], actions=[Action.STAY])
    assert len(Sensor) > 1, "capability selection mutated the Sensor enum"

    unrestricted = World(config=config, n_organisms=4)
    assert set(unrestricted.config.enabled_sensors) == set(Sensor)


# ----------------------------------------------------------------------------
# Driving a generation step by step
# ----------------------------------------------------------------------------


def test_iter_generation_yields_every_step_then_returns_survivors(config: Settings) -> None:
    """The generator form is what lets the web server stream a run."""
    world = World(config=config)
    generation = world.iter_generation()

    steps = []
    try:
        while True:
            steps.append(next(generation))
    except StopIteration as finished:
        survivors = finished.value

    assert steps == list(range(config.steps_per_generation))
    assert 0 <= survivors <= config.n_organisms
    assert world.generation == 1


def test_abandoning_a_generation_does_not_select_or_repopulate(config: Settings) -> None:
    """
    A run stopped halfway must leave the world alone rather than breeding from it.

    The web UI abandons the generator whenever the Stop button is pressed, or a
    new run replaces the current one.
    """
    world = World(config=config)
    original = [org.genome for org in world.organisms]

    generation = world.iter_generation()
    for _ in range(5):
        next(generation)
    generation.close()

    assert world.generation == 0, "an abandoned generation should not count"
    assert [org.genome for org in world.organisms] == original, "abandoned run bred anyway"


def test_run_generation_matches_the_generator(config: Settings) -> None:
    """`run_generation` is a thin wrapper, and must stay equivalent to it."""
    seen: list[int] = []
    world = World(config=config)
    survivors = world.run_generation(on_step=lambda _world, step: seen.append(step))

    assert seen == list(range(config.steps_per_generation))
    assert 0 <= survivors <= config.n_organisms
    assert world.generation == 1


def test_evolution_improves_survival() -> None:
    """
    The whole point: survival must climb well above where random genomes start.

    Slow-ish, but a simulation that no longer evolves is the one failure that
    every other test here would happily pass through.
    """
    config = replace(Settings(), n_organisms=150, steps_per_generation=100)
    world = World(config=config, objective="left")

    first = world.run_generation()
    for _ in range(19):
        last = world.run_generation()

    assert last > first, f"no improvement: started at {first}, ended at {last}"
    assert last / config.n_organisms > 0.6, f"only {last}/{config.n_organisms} survived"


# ----------------------------------------------------------------------------
# Saving, loading and drawing
# ----------------------------------------------------------------------------


def test_genome_survives_a_save_and_load(config: Settings, tmp_path: Path) -> None:
    """A kept creature is worthless if reloading it changes its behaviour."""
    original = brain_utils.random_genome(config)
    path = inspect_utils.save_genome(original, config, tmp_path / "creature.json")
    restored = inspect_utils.load_genome(path)

    assert restored == original
    assert Brain(restored, config).describe() == Brain(original, config).describe()


def test_saved_genomes_are_written_by_name(config: Settings, tmp_path: Path) -> None:
    """
    Names, not enum values, so a file stays valid when new senses are added.

    A genome saved by index would silently mean something different after
    anyone inserted a sensor in the middle of the enum.
    """
    genome = (Gene(Source.SENSOR, Sensor.AGE, Sink.ACTION, Action.MOVE_X, 1.0),)
    path = inspect_utils.save_genome(genome, config, tmp_path / "named.json")
    text = path.read_text(encoding="utf-8")
    assert "age" in text
    assert "move_x" in text


def test_a_genome_from_the_future_is_refused(config: Settings, tmp_path: Path) -> None:
    """An unreadable file should say so rather than load as nonsense."""
    path = tmp_path / "future.json"
    path.write_text('{"version": 99, "genes": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported genome format"):
        inspect_utils.load_genome(path)


def test_brain_diagram_draws_without_a_display(config: Settings) -> None:
    """The wiring diagram is only useful if it renders for any genome."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    brain = Brain(brain_utils.random_genome(config), config)
    axes = inspect_utils.draw_brain(brain, config)
    assert axes.get_title()
    plt.close("all")

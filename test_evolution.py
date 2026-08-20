"""
Checks for metabolism, population dynamics, reproduction and observation.

These are the mechanics that decide whether a population lives or dies, so the
invariants worth pinning are mostly about *pressure*: that cost is charged, that
the population can genuinely collapse, and that a strategy which cannot find
partners really does fail rather than quietly falling back to cloning.
"""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest

import population_stats
import reproduction
from brain_utils import Gene, Sink, Source
from capability_utils import Action, Sensor
from organism import Organism, World
from reproduction import Asexual, Sexual
from settings import Settings


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    random.seed(0)
    np.random.seed(0)


@pytest.fixture
def config() -> Settings:
    return replace(Settings(), n_organisms=40, steps_per_generation=30)


def straight_line_genome() -> tuple[Gene, ...]:
    """A minimal brain: one sense, one action, so upkeep is predictable."""
    return (Gene(Source.SENSOR, Sensor.BIAS, Sink.ACTION, Action.MOVE_X, 4.0),)


# ----------------------------------------------------------------------------
# Metabolism
# ----------------------------------------------------------------------------


def test_upkeep_scales_with_distinct_senses_not_gene_count(config: Settings) -> None:
    """
    Wiring the same sense twice is free; reaching for a new one is not.

    This is what makes the cost a pressure on *breadth* of perception rather
    than a flat tax on genome size.
    """
    twice_the_same = Organism(
        config,
        genome=(
            Gene(Source.SENSOR, Sensor.BIAS, Sink.ACTION, Action.MOVE_X, 1.0),
            Gene(Source.SENSOR, Sensor.BIAS, Sink.ACTION, Action.MOVE_Y, 1.0),
        ),
    )
    two_different = Organism(
        config,
        genome=(
            Gene(Source.SENSOR, Sensor.BIAS, Sink.ACTION, Action.MOVE_X, 1.0),
            Gene(Source.SENSOR, Sensor.AGE, Sink.ACTION, Action.MOVE_Y, 1.0),
        ),
    )
    assert two_different.upkeep > twice_the_same.upkeep
    assert twice_the_same.upkeep == pytest.approx(config.metabolism + config.sense_cost)


def test_a_costly_brain_starves_before_a_lean_one(config: Settings) -> None:
    """The whole point of metabolism: complexity has to earn its keep."""
    lean = Organism(config, genome=straight_line_genome())
    bloated = Organism(
        config,
        genome=tuple(
            Gene(Source.SENSOR, sensor, Sink.ACTION, Action.MOVE_X, 1.0) for sensor in Sensor
        ),
    )
    assert bloated.upkeep > lean.upkeep

    world = World(config=config, n_organisms=1)
    world.organisms = [lean, bloated]
    world.reset_grid(world.organisms)
    for _ in range(config.steps_per_generation):
        world.update_organisms()
    assert bloated.energy < lean.energy


def test_starvation_kills_and_frees_the_cell(config: Settings) -> None:
    """A starved organism must leave play properly, not linger as a blocker."""
    starving = replace(config, initial_energy=0.5, metabolism=1.0)
    world = World(config=starving, n_organisms=1)
    org = world.organisms[0]
    world.relocate(org, 40, 40)

    org.act(world)
    assert not org.alive
    assert not world.is_occupied(40, 40)


def test_metabolism_can_be_switched_off(config: Settings) -> None:
    """With energy disabled nobody starves, however expensive the brain."""
    free = replace(config, energy_enabled=False, initial_energy=0.1, metabolism=10.0)
    world = World(config=free)
    for _ in range(20):
        world.update_organisms()
    assert all(org.alive for org in world.organisms)


def test_the_energy_sense_reports_the_tank(config: Settings) -> None:
    world = World(config=config, n_organisms=1)
    org = world.organisms[0]
    assert org.perceive(world)[Sensor.ENERGY] == pytest.approx(1.0)

    org.energy = config.initial_energy / 2
    assert org.perceive(world)[Sensor.ENERGY] == pytest.approx(0.5)

    org.energy = -5.0
    assert org.perceive(world)[Sensor.ENERGY] == 0.0


# ----------------------------------------------------------------------------
# Population dynamics
# ----------------------------------------------------------------------------


def test_population_grows_and_shrinks_with_survival(config: Settings) -> None:
    """
    The knife's edge: `offspring_per_survivor` sets the replacement rate.

    At 2.0, half the population surviving exactly replaces it -- so this is the
    number that decides whether a run recovers or slides to extinction.
    """
    world = World(config=replace(config, offspring_per_survivor=2.0, carrying_capacity=1000))
    survivors = world.organisms[:10]
    world.reproduce_organisms(survivors)
    assert world.population == 20


def test_carrying_capacity_caps_a_boom(config: Settings) -> None:
    world = World(config=replace(config, offspring_per_survivor=10.0, carrying_capacity=25))
    world.reproduce_organisms(world.organisms[:20])
    assert world.population == 25


def test_no_survivors_means_extinction(config: Settings) -> None:
    """
    The population is no longer refilled from nowhere, so a wipe-out is final.

    This replaces the old reseeding behaviour: a run that dies out has to say
    so rather than quietly starting again with random genomes.
    """
    world = World(config=config)
    world.reproduce_organisms([])

    assert world.extinct
    assert world.population == 0
    assert int(world.occupancy.sum()) == 0
    assert world.run_generation() == 0, "an extinct world should not keep running"


def test_a_population_can_die_out_over_a_run() -> None:
    """End to end: harsh enough conditions really do end a run."""
    doomed = replace(
        Settings(),
        n_organisms=30,
        steps_per_generation=40,
        survival_zone_fraction=0.02,
        offspring_per_survivor=1.0,
    )
    world = World(config=doomed, objective="left")
    for _ in range(40):
        world.run_generation()
        if world.extinct:
            break
    assert world.extinct, "these conditions should not be survivable"


# ----------------------------------------------------------------------------
# The shrinking zone
# ----------------------------------------------------------------------------


def test_the_zone_contracts_as_generations_pass(config: Settings) -> None:
    world = World(config=replace(config, zone_shrink_per_generation=0.05), objective="left")

    start = world.zone_fraction
    world.generation = 20
    assert world.zone_fraction < start

    zone = world.objective.zones(world)[0].mask.sum()
    world.generation = 0
    assert world.objective.zones(world)[0].mask.sum() > zone, "the drawn zone must shrink too"


def test_the_zone_never_shrinks_past_its_floor(config: Settings) -> None:
    """A target that vanishes entirely would make the run unwinnable by accident."""
    world = World(config=replace(config, zone_shrink_per_generation=0.5))
    world.generation = 500
    assert world.zone_fraction == pytest.approx(config.min_zone_fraction)


def test_a_fixed_zone_does_not_move(config: Settings) -> None:
    world = World(config=replace(config, zone_shrink_per_generation=0.0))
    before = world.zone_fraction
    world.generation = 99
    assert world.zone_fraction == before


# ----------------------------------------------------------------------------
# Reproduction
# ----------------------------------------------------------------------------


def test_asexual_offspring_are_mutated_clones(config: Settings) -> None:
    quiet = replace(config, point_mutation_rate=0.0)
    parent = Organism(quiet, genome=straight_line_genome())
    child = Asexual().offspring([parent], quiet)
    assert child.genome == parent.genome
    assert child.lineage == parent.lineage


def test_crossover_takes_every_gene_from_one_parent_or_the_other(config: Settings) -> None:
    """A child must be a recombination of its parents, never an invention."""
    quiet = replace(config, point_mutation_rate=0.0, n_genes=12)
    mother = Organism(quiet)
    father = Organism(quiet)

    child_genome = Sexual().combine([mother, father], quiet)
    assert len(child_genome) == len(mother.genome)
    for index, gene in enumerate(child_genome):
        assert gene in (mother.genome[index], father.genome[index])


def test_crossover_actually_mixes_the_parents(config: Settings) -> None:
    """Uniform crossover should draw from both sides, not copy one parent."""
    quiet = replace(config, point_mutation_rate=0.0, n_genes=40)
    mother, father = Organism(quiet), Organism(quiet)

    from_father = 0
    for _ in range(20):
        child = Sexual().combine([mother, father], quiet)
        from_father += sum(1 for index, gene in enumerate(child) if gene == father.genome[index])
    share = from_father / (20 * 40)
    assert 0.3 < share < 0.7, f"crossover looks lopsided: {share:.0%} from the father"


def test_sexual_reproduction_needs_a_partner_in_range(config: Settings) -> None:
    """
    A survivor with nobody nearby leaves no offspring at all.

    This is the whole point of the sexual strategy: reaching the zone is not
    enough if you arrive alone.
    """
    close = replace(config, mating_radius=5)
    world = World(config=close, n_organisms=2, strategy="sexual")
    first, second = world.organisms

    world.relocate(first, 5, 5)
    world.relocate(second, 90, 90)
    assert Sexual().breeding_pairs([first, second], close) == []

    world.relocate(second, 8, 8)
    assert len(Sexual().breeding_pairs([first, second], close)) == 1


def test_a_lone_survivor_ends_a_sexual_run(config: Settings) -> None:
    """One creature cannot repopulate the world on its own."""
    world = World(config=config, strategy="sexual")
    world.reproduce_organisms([world.organisms[0]])
    assert world.extinct


def test_a_lone_survivor_can_repopulate_an_asexual_run(config: Settings) -> None:
    """The contrast that makes the sexual case meaningful."""
    world = World(config=config, strategy="asexual")
    world.reproduce_organisms([world.organisms[0]])
    assert not world.extinct
    assert world.population > 0


def test_pairing_is_monogamous(config: Settings) -> None:
    """Each survivor pairs at most once, so a crowd cannot all breed with one creature."""
    world = World(config=config, n_organisms=6, strategy="sexual")
    for org in world.organisms:
        world.relocate(org, 50 + world.organisms.index(org), 50)

    pairs = Sexual().breeding_pairs(world.organisms, config)
    paired = [org for pair in pairs for org in pair]
    assert len(paired) == len(set(map(id, paired))), "an organism bred twice"


def test_unknown_strategy_is_reported_clearly() -> None:
    with pytest.raises(ValueError, match="unknown reproduction strategy"):
        reproduction.resolve("budding")


@pytest.mark.parametrize("strategy", list(reproduction.STRATEGIES))
def test_every_strategy_can_run_a_generation(strategy: str, config: Settings) -> None:
    world = World(config=replace(config, n_organisms=80), objective="left", strategy=strategy)
    world.run_generation()
    assert world.extinct or world.population > 0


# ----------------------------------------------------------------------------
# Lineage
# ----------------------------------------------------------------------------


def test_founders_each_start_their_own_line(config: Settings) -> None:
    world = World(config=config)
    assert len({org.lineage for org in world.organisms}) == config.n_organisms


def test_children_inherit_a_parent_line(config: Settings) -> None:
    world = World(config=config, strategy="asexual")
    parent = world.organisms[0]
    world.reproduce_organisms([parent])
    assert all(org.lineage == parent.lineage for org in world.organisms)


def test_a_sexual_child_takes_one_parent_line(config: Settings) -> None:
    """With two parents the line is a coin toss, never a new invented one."""
    mother, father = Organism(config), Organism(config)
    mother.lineage, father.lineage = 11, 22

    lines = {Sexual().offspring([mother, father], config).lineage for _ in range(40)}
    assert lines <= {11, 22}
    assert len(lines) == 2, "one parent's line is never being passed on"


def test_lineages_collapse_as_a_run_proceeds(config: Settings) -> None:
    """
    Selection should concentrate the gene pool, not preserve every founder.

    Run under gentle conditions on purpose: the point is to watch lineages
    narrow, which needs a population that survives long enough to do it.
    """
    gentle = replace(
        config,
        n_organisms=120,
        energy_enabled=False,
        survival_zone_fraction=0.25,
    )
    world = World(config=gentle, objective="left")
    before = population_stats.lineages(world)["alive"]
    assert before == 120, "founders should start as 120 separate lines"

    for _ in range(6):
        world.run_generation()
    assert not world.extinct, "gentle conditions should not wipe the population out"
    assert population_stats.lineages(world)["alive"] < before


# ----------------------------------------------------------------------------
# Observing the population
# ----------------------------------------------------------------------------


def test_expression_covers_every_capability(config: Settings) -> None:
    """The chart needs a row per capability, including the ones nobody uses."""
    world = World(config=config)
    stats = population_stats.expression(world)

    assert [item["label"] for item in stats["sensors"]] == [str(s) for s in Sensor]
    assert [item["label"] for item in stats["actions"]] == [str(a) for a in Action]
    for item in stats["sensors"] + stats["actions"]:
        assert 0.0 <= item["share"] <= 1.0


def test_expression_reports_what_the_population_actually_wires(config: Settings) -> None:
    """A population wired to one sense must read as exactly that."""
    world = World(config=config)
    for org in world.organisms:
        org.genome = straight_line_genome()
        org.brain = type(org.brain)(org.genome, config)

    stats = population_stats.expression(world)
    shares = {item["label"]: item["share"] for item in stats["sensors"]}
    assert shares["bias"] == 1.0
    assert all(share == 0.0 for label, share in shares.items() if label != "bias")
    assert stats["mean_senses_used"] == pytest.approx(1.0)


def test_expression_survives_an_empty_population(config: Settings) -> None:
    """An extinct run still has to render without blowing up the UI."""
    world = World(config=config)
    world.reproduce_organisms([])
    stats = population_stats.expression(world)
    assert stats["population"] == 0
    assert len(stats["sensors"]) == len(Sensor)


def test_disabled_capabilities_read_as_unused(config: Settings) -> None:
    """The expression view should show a switched-off sense sitting at zero."""
    limited = config.with_capabilities(sensors=[Sensor.BIAS], actions=[Action.MOVE_X])
    world = World(config=limited)
    stats = population_stats.expression(world)
    shares = {item["label"]: item["share"] for item in stats["sensors"]}
    assert shares["bias"] == 1.0
    assert shares["pheromone_here"] == 0.0


def test_the_text_summary_names_what_is_used(config: Settings) -> None:
    world = World(config=config)
    text = population_stats.summarise(world)
    assert "population" in text
    assert "lineages" in text

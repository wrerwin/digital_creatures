"""
Sanity checks for the simulation. Run with:

    python tests.py

These are deliberately about invariants rather than exact values -- the whole
system is stochastic, so anything that asserts a specific outcome would be
flaky. What must always hold is that organisms stay on the grid, never share a
cell, and that mutation only ever produces genes that can be wired up.
"""

import random
import sys
import traceback

import numpy as np

import brain_utils
import capability_utils
import settings
from organism import CRITERIA, organism, world

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# ----------------------------------------------------------------------------
# The world and its occupancy grid
# ----------------------------------------------------------------------------

@test
def organisms_stay_in_bounds():
    '''Organisms must never walk off the grid, however long the run.'''
    sim = world(n_organisms=60)
    for _ in range(300):
        sim.update_organisms()
        for org in sim.organism_states:
            assert sim.in_bounds(org.x, org.y), \
                'organism escaped to ({}, {})'.format(org.x, org.y)


@test
def organisms_do_not_get_stuck_at_the_walls():
    '''
    An organism that reaches a wall must still be able to move along it.

    This is the failure the original move() had: its bounds check froze any
    organism that stepped onto the boundary.
    '''
    sim = world(n_organisms=1)
    org = sim.organism_states[0]
    sim.relocate(org, 0, sim.height // 2)

    # Drive it straight into the west wall, then along the wall.
    org.move(-1, 0, sim)
    assert org.x == 0, 'organism should be held at the wall, not pushed through'

    org.move(0, 1, sim)
    assert org.y == sim.height // 2 + 1, 'organism at a wall should still move along it'


@test
def one_organism_per_cell():
    '''Occupancy must stay exclusive, and the grid must match the population.'''
    sim = world(n_organisms=80)
    for _ in range(120):
        sim.update_organisms()

    cells = set((org.x, org.y) for org in sim.organism_states)
    assert len(cells) == len(sim.organism_states), 'two organisms share a cell'
    assert int(sim.occupancy.sum()) == len(sim.organism_states), \
        'occupancy grid disagrees with the population'


@test
def blocked_moves_are_refused():
    '''An organism cannot step onto a cell another organism already holds.'''
    sim = world(n_organisms=2)
    first, second = sim.organism_states
    sim.relocate(first, 10, 10)
    sim.relocate(second, 11, 10)

    first.move(1, 0, sim)
    assert (first.x, first.y) == (10, 10), 'organism moved into an occupied cell'


@test
def population_density_reads_neighbours_not_self():
    '''An organism alone in an empty world should read zero density.'''
    sim = world(n_organisms=1)
    org = sim.organism_states[0]
    sim.relocate(org, 50, 50)
    assert sim.population_density(50, 50, 4) == 0.0, 'an organism sensed itself'

    crowded = world(n_organisms=1)
    crowded.occupancy[48:53, 48:53] = 1
    assert crowded.population_density(50, 50, 2) == 1.0, \
        'a full neighbourhood should read as fully dense'


# ----------------------------------------------------------------------------
# Genomes and brains
# ----------------------------------------------------------------------------

@test
def mutation_only_produces_valid_genes():
    '''
    Every gene must point at endpoints that exist.

    Mutation can flip an endpoint's *kind*, and each kind has a different id
    range, so this is the property most likely to break when the sensor or
    action lists change.
    '''
    genome = brain_utils.random_genome()
    for _ in range(400):
        genome = brain_utils.mutate(genome, rate=0.5)
        for gene in genome:
            if gene.source_type == brain_utils.SENSOR:
                assert 0 <= gene.source_id < capability_utils.N_SENSORS
            else:
                assert 0 <= gene.source_id < settings.n_inner_neurons

            if gene.sink_type == brain_utils.INNER:
                assert 0 <= gene.sink_id < settings.n_inner_neurons
            else:
                assert 0 <= gene.sink_id < capability_utils.N_ACTIONS

            assert abs(gene.weight) <= settings.max_weight


@test
def mutation_does_not_touch_the_parent():
    '''Reproduction must copy the genome, not edit the parent's in place.'''
    parent = brain_utils.random_genome()
    before = [(g.source_type, g.source_id, g.sink_type, g.sink_id, g.weight)
              for g in parent]
    brain_utils.mutate(parent, rate=1.0)
    after = [(g.source_type, g.source_id, g.sink_type, g.sink_id, g.weight)
             for g in parent]
    assert before == after, 'mutation modified the parent genome'


@test
def brains_produce_one_bounded_level_per_action():
    '''think() must return a usable level for every action, every time.'''
    sim = world(n_organisms=30)
    for _ in range(20):
        for org in sim.organism_states:
            levels = org.brain.think(org, sim)
            assert len(levels) == capability_utils.N_ACTIONS
            for level in levels:
                assert -1.0 <= level <= 1.0, 'action level out of range: {}'.format(level)
        sim.update_organisms()


@test
def brains_only_read_the_sensors_they_are_wired_to():
    '''The lazy sensor set must match what the genome actually references.'''
    genome = brain_utils.random_genome()
    brain = brain_utils.Brain(genome)
    expected = set(g.source_id for g in genome if g.source_type == brain_utils.SENSOR)
    assert set(brain.needed_sensors) == expected


@test
def perceive_reports_every_sensor():
    '''The debugging view must cover the full sensor list, in range.'''
    sim = world(n_organisms=5)
    knowledge = sim.organism_states[0].perceive(sim)
    assert set(knowledge) == set(capability_utils.SENSOR_NAMES)
    for name, value in knowledge.items():
        assert -1.0 <= value <= 1.0, '{} out of range: {}'.format(name, value)


# ----------------------------------------------------------------------------
# Selection and reproduction
# ----------------------------------------------------------------------------

@test
def a_generation_preserves_population_size():
    '''However selection goes, the next generation must be full and placed.'''
    sim = world(n_organisms=50)
    for _ in range(3):
        sim.run_generation()
        assert len(sim.organism_states) == 50
        for org in sim.organism_states:
            assert org.x is not None and org.y is not None, 'organism was never placed'
            assert org.age == 0, 'a new generation should start at age zero'


@test
def extinction_reseeds_rather_than_ending_the_run():
    '''With no survivors the world must refill itself instead of emptying.'''
    sim = world(n_organisms=25, survival_criterion=lambda org, w: False)
    sim.run_generation()
    assert len(sim.organism_states) == 25


@test
def survivors_satisfy_the_criterion():
    '''Selection must return exactly the organisms inside the survival zone.'''
    for name, criterion in CRITERIA.items():
        sim = world(n_organisms=40, survival_criterion=criterion)
        for _ in range(30):
            sim.update_organisms()
        survivors = sim.select_survivors()
        for org in survivors:
            assert criterion(org, sim), '{}: selected an organism outside the zone'.format(name)
        for org in sim.organism_states:
            if org not in survivors:
                assert not criterion(org, sim), '{}: missed a qualifying organism'.format(name)


@test
def a_child_inherits_its_parents_wiring():
    '''At a zero mutation rate, offspring must be exact copies.'''
    parent = organism()
    child = organism(genome=brain_utils.mutate(parent.genome, rate=0.0))
    assert child.brain.describe() == parent.brain.describe()


# ----------------------------------------------------------------------------

def main():
    random.seed(0)
    np.random.seed(0)

    failures = 0
    for fn in TESTS:
        try:
            fn()
        except Exception:
            failures += 1
            print('FAIL  {}'.format(fn.__name__))
            traceback.print_exc()
        else:
            print('ok    {}'.format(fn.__name__))

    print('\n{} passed, {} failed'.format(len(TESTS) - failures, failures))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())

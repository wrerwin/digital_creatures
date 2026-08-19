"""
The organisms and the world they live in.

A generation runs for `settings.steps_per_generation` timesteps. At the end of
it a survival criterion decides who reproduces; everybody else dies. Survivors
are cloned with mutation until the population is full again, and the next
generation starts from scratch on an empty grid.
"""

import random

import numpy as np

import brain_utils
import capability_utils
import settings

# The eight ways a step can point, used when an organism moves at random.
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1), (0, 1),
              (1, -1), (1, 0), (1, 1)]


class organism(object):
    '''
    One creature: a position on the grid, a genome, and the brain it builds.

    An organism does not know how to find anything out for itself -- every
    sense it has comes from capability_utils, and which of those senses it
    actually consults is decided by its genome.
    '''

    def __init__(self, genome=None, name=None):
        self.name = name
        self.genome = genome if genome is not None else brain_utils.random_genome()
        self.brain = brain_utils.Brain(self.genome)

        # Filled in by world.place(); an unplaced organism has no position.
        self.x = None
        self.y = None

        self.last_dx = 0
        self.last_dy = 0
        self.age = 0

    # ------------------------------------------------------------------
    # Perceiving and acting
    # ------------------------------------------------------------------

    def perceive(self, world):
        '''
        Every sense this organism has, as a name -> value dict.

        This is for inspection and debugging. The brain reads sensors lazily
        during think(), evaluating only the ones its genome refers to.
        '''
        return dict(zip(capability_utils.SENSOR_NAMES,
                        capability_utils.read_sensors(self, world)))

    def act(self, world):
        '''
        Run the brain for one timestep and resolve its output into a step.

        The action neurons compete rather than take turns: their levels are
        summed into an urge along each axis, which is then converted into an
        actual grid step probabilistically, so a weak urge moves the organism
        sometimes and a strong one moves it almost always.
        '''
        levels = self.brain.think(self, world)
        index = capability_utils.ACTION_INDEX

        urge_x = levels[index['move_x']]
        urge_y = levels[index['move_y']]

        forward = levels[index['move_forward']]
        urge_x += forward * self.last_dx
        urge_y += forward * self.last_dy

        wander = levels[index['move_random']]
        if wander != 0.0:
            dx, dy = random.choice(DIRECTIONS)
            urge_x += wander * dx
            urge_y += wander * dy

        # A positive 'stay' level damps whatever the other neurons wanted.
        stillness = max(0.0, levels[index['stay']])
        urge_x *= (1.0 - stillness)
        urge_y *= (1.0 - stillness)

        self.move(_urge_to_step(urge_x), _urge_to_step(urge_y), world)
        self.age += 1

    def move(self, dx, dy, world):
        '''
        Try to step by (dx, dy), refusing moves into walls or occupied cells.

        A blocked diagonal falls back to whichever single axis is still open,
        which lets organisms slide along a wall instead of sticking to it.
        '''
        if dx == 0 and dy == 0:
            self.last_dx = 0
            self.last_dy = 0
            return

        for step_x, step_y in ((dx, dy), (dx, 0), (0, dy)):
            if step_x == 0 and step_y == 0:
                continue
            if world.can_move_to(self.x + step_x, self.y + step_y):
                world.relocate(self, self.x + step_x, self.y + step_y)
                self.last_dx = step_x
                self.last_dy = step_y
                return

        # Fully boxed in: keep the heading so 'blocked_forward' can report it.
        self.last_dx = dx
        self.last_dy = dy

    # ------------------------------------------------------------------
    # Reproduction
    # ------------------------------------------------------------------

    def reproduce(self):
        '''
        Produce one offspring: this organism's genome, copied with mutations.

        Reproduction is asexual, so all new variation comes from mutation.
        The child is unplaced -- the world decides where it starts.
        '''
        return organism(genome=brain_utils.mutate(self.genome))

    def __repr__(self):
        return '<organism at ({}, {})>'.format(self.x, self.y)


def _urge_to_step(urge):
    '''
    Turn a continuous urge into one of -1, 0, +1.

    The magnitude is read as a probability, so behaviour stays stochastic:
    an urge of 0.3 produces a step about a third of the time.
    '''
    urge = max(-1.0, min(1.0, urge))
    if random.random() >= abs(urge):
        return 0
    return 1 if urge > 0 else -1


class world(object):
    '''
    The grid, the population living on it, and the generational cycle.

    Occupancy is kept in a numpy array alongside the organism list: the list is
    what we iterate over, the array is what makes "is that cell taken?" and
    "how crowded is it here?" cheap enough to ask on every timestep.
    '''

    def __init__(self, n_organisms=None, survival_criterion=None):
        if n_organisms is None:
            n_organisms = settings.n_organisms
        self.n_organisms = n_organisms
        self.survival_criterion = survival_criterion or left_edge_criterion

        self.width = settings.x_max - settings.x_min
        self.height = settings.y_max - settings.y_min

        self.generation = 0
        self.step = 0
        self.occupancy = np.zeros((self.width, self.height), dtype=np.int8)
        self.organism_states = self.build_initial_config()

    # ------------------------------------------------------------------
    # Setting up
    # ------------------------------------------------------------------

    def build_initial_config(self):
        '''Create the founding population, each with a fully random genome.'''
        organisms = [organism() for _ in range(self.n_organisms)]
        self.reset_grid(organisms)
        return organisms

    def reset_grid(self, organisms):
        '''Clear the grid and scatter the given organisms over empty cells.'''
        self.occupancy[:, :] = 0
        self.step = 0
        for org in organisms:
            org.x = None
            org.y = None
            org.last_dx = 0
            org.last_dy = 0
            org.age = 0
            org.brain.reset()
            self.place(org, *self.random_empty_cell())

    def random_empty_cell(self):
        '''Pick an unoccupied cell, falling back to a scan if the grid is full-ish.'''
        for _ in range(100):
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            if not self.occupancy[x, y]:
                return x, y

        empty = np.argwhere(self.occupancy == 0)
        if len(empty) == 0:
            raise RuntimeError('no empty cells left: population exceeds grid size')
        x, y = empty[random.randrange(len(empty))]
        return int(x), int(y)

    # ------------------------------------------------------------------
    # Grid queries, used by the sensors
    # ------------------------------------------------------------------

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_occupied(self, x, y):
        return bool(self.occupancy[x, y])

    def can_move_to(self, x, y):
        return self.in_bounds(x, y) and not self.occupancy[x, y]

    def population_density(self, x, y, radius):
        '''
        Fraction of the cells around (x, y) that hold another organism.

        The neighbourhood is clipped at the walls, so an organism in a corner
        does not read as lonely just because part of its window is off-grid.
        '''
        x0 = max(0, x - radius)
        x1 = min(self.width, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(self.height, y + radius + 1)

        window = self.occupancy[x0:x1, y0:y1]
        neighbours = int(window.sum()) - 1  # discount the organism itself
        cells = window.size - 1
        if cells <= 0:
            return 0.0
        return neighbours / cells

    def place(self, org, x, y):
        org.x = x
        org.y = y
        self.occupancy[x, y] = 1

    def relocate(self, org, x, y):
        self.occupancy[org.x, org.y] = 0
        self.occupancy[x, y] = 1
        org.x = x
        org.y = y

    # ------------------------------------------------------------------
    # Running time
    # ------------------------------------------------------------------

    def update_organisms(self):
        '''
        Advance every organism by one timestep.

        The order is shuffled each step so that no organism gets a permanent
        advantage in claiming contested cells.
        '''
        order = list(self.organism_states)
        random.shuffle(order)
        for org in order:
            org.act(self)
        self.step += 1

    def run_generation(self, on_step=None):
        '''
        Run one full generation, then select and repopulate.

        `on_step` is called with (world, step) after each timestep, which is
        how execute.py draws the animation without the world knowing about it.

        Returns the number of organisms that met the survival criterion.
        '''
        for step in range(settings.steps_per_generation):
            self.update_organisms()
            if on_step is not None:
                on_step(self, step)

        survivors = self.select_survivors()
        self.reproduce_organisms(survivors)
        self.generation += 1
        return len(survivors)

    def select_survivors(self):
        '''The organisms that satisfy the survival criterion. Everyone else dies.'''
        return [org for org in self.organism_states if self.survival_criterion(org, self)]

    def reproduce_organisms(self, survivors):
        '''
        Refill the population from the survivors and reset the grid.

        Parents are drawn at random with replacement, so a lone survivor can
        found an entire generation. With no survivors at all the run would end,
        so the world reseeds itself with fresh random genomes instead.
        '''
        if not survivors:
            self.organism_states = [organism() for _ in range(self.n_organisms)]
            self.reset_grid(self.organism_states)
            return

        children = [random.choice(survivors).reproduce()
                    for _ in range(self.n_organisms)]
        self.organism_states = children
        self.reset_grid(children)

    def positions(self):
        '''Current x and y arrays for the whole population, for plotting.'''
        xs = np.fromiter((org.x for org in self.organism_states), dtype=int,
                         count=len(self.organism_states))
        ys = np.fromiter((org.y for org in self.organism_states), dtype=int,
                         count=len(self.organism_states))
        return xs, ys


# ----------------------------------------------------------------------------
# Survival criteria
# ----------------------------------------------------------------------------
#
# A criterion takes (organism, world) and returns True if that organism gets to
# reproduce. This is the entire selection pressure -- swapping the criterion is
# the main dial for changing what evolves.

def left_edge_criterion(org, world):
    '''Survive by ending the generation in a band along the west wall.'''
    return org.x < world.width * settings.survival_zone_fraction


def right_edge_criterion(org, world):
    '''Survive by ending the generation in a band along the east wall.'''
    return org.x >= world.width * (1 - settings.survival_zone_fraction)


def centre_criterion(org, world):
    '''Survive by ending the generation inside a circle at the middle of the world.'''
    radius = world.width * settings.survival_zone_fraction
    dx = org.x - world.width / 2
    dy = org.y - world.height / 2
    return (dx * dx + dy * dy) <= radius * radius


def corners_criterion(org, world):
    '''Survive by ending the generation in any of the four corners.'''
    span = world.width * settings.survival_zone_fraction
    near_x = org.x < span or org.x >= world.width - span
    near_y = org.y < span or org.y >= world.height - span
    return near_x and near_y


CRITERIA = {
    'left': left_edge_criterion,
    'right': right_edge_criterion,
    'centre': centre_criterion,
    'corners': corners_criterion,
}

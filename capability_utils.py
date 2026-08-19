"""
Defines what an organism is able to perceive and what it is able to do.

These two lists are the organism's interface to the world. A genome wires
sensors to actions (possibly via inner neurons), so adding an entry here
immediately widens the space of behaviours evolution can discover -- no
other file needs to change.

Sensor functions take (organism, world) and return a float. Values are kept
in [-1, 1] so that no single sensor dominates the weighted sums in the brain.
"""

from numpy.random import uniform

import settings

# Radius (in cells) of the neighbourhood an organism can feel around itself.
DENSITY_RADIUS = 4


# ----------------------------------------------------------------------------
# Sensors
# ----------------------------------------------------------------------------

def sense_x_position(org, world):
    '''Where the organism sits along the x axis, 0 at the west wall, 1 at the east.'''
    return org.x / (settings.x_max - 1)


def sense_y_position(org, world):
    '''Where the organism sits along the y axis, 0 at the south wall, 1 at the north.'''
    return org.y / (settings.y_max - 1)


def sense_border_distance(org, world):
    '''0 when pressed against the nearest wall, 1 when in the middle of the world.'''
    to_wall = min(org.x,
                  org.y,
                  settings.x_max - 1 - org.x,
                  settings.y_max - 1 - org.y)
    half_span = min(settings.x_max, settings.y_max) / 2
    return min(to_wall / half_span, 1.0)


def sense_population_density(org, world):
    '''Fraction of nearby cells that are occupied by another organism.'''
    return world.population_density(org.x, org.y, DENSITY_RADIUS)


def sense_blocked_forward(org, world):
    '''1 if the cell the organism last moved toward is now a wall or another organism.'''
    if org.last_dx == 0 and org.last_dy == 0:
        return 0.0
    ahead_x = org.x + org.last_dx
    ahead_y = org.y + org.last_dy
    if not world.in_bounds(ahead_x, ahead_y) or world.is_occupied(ahead_x, ahead_y):
        return 1.0
    return 0.0


def sense_last_move_x(org, world):
    '''The x component of the organism's previous step.'''
    return float(org.last_dx)


def sense_last_move_y(org, world):
    '''The y component of the organism's previous step.'''
    return float(org.last_dy)


def sense_age(org, world):
    '''0 at the start of a generation, 1 at the end of it.'''
    return org.age / settings.steps_per_generation


def sense_random(org, world):
    '''Noise, so that evolution has a source of unpredictability to draw on.'''
    return float(uniform(-1, 1))


def sense_bias(org, world):
    '''Constant input, which lets a connection act as a fixed drive.'''
    return 1.0


SENSORS = [
    ('x_position', sense_x_position),
    ('y_position', sense_y_position),
    ('border_distance', sense_border_distance),
    ('population_density', sense_population_density),
    ('blocked_forward', sense_blocked_forward),
    ('last_move_x', sense_last_move_x),
    ('last_move_y', sense_last_move_y),
    ('age', sense_age),
    ('random', sense_random),
    ('bias', sense_bias),
]

SENSOR_NAMES = [name for name, _ in SENSORS]
N_SENSORS = len(SENSORS)


def read_sensors(org, world):
    '''Evaluate every sensor for one organism, returning a list of floats.'''
    return [sensor(org, world) for _, sensor in SENSORS]


# ----------------------------------------------------------------------------
# Actions
# ----------------------------------------------------------------------------
#
# Action neurons do not move the organism directly. Each one produces a level
# in [-1, 1] (the brain squashes it with tanh) and organism.act() resolves the
# competing levels into a single step on the grid.

ACTIONS = [
    'move_x',        # signed drive along x: positive is east, negative is west
    'move_y',        # signed drive along y: positive is north, negative is south
    'move_forward',  # keep going the way you last went
    'move_random',   # take an arbitrary step
    'stay',          # suppress movement
]

ACTION_NAMES = list(ACTIONS)
N_ACTIONS = len(ACTIONS)

# Index lookups, so organism.act() can name actions instead of counting.
ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS)}

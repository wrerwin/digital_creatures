"""
Genomes and the brains they build.

A genome is a fixed-length list of genes. Each gene describes one connection:

    source (a sensor, or an inner neuron)  --weight-->  sink (an inner neuron, or an action)

Because the *endpoints* are part of the gene rather than fixed in advance, a
mutation can rewire an organism onto a sense it never used before, or hand
control of an action to a different part of its brain. The topology evolves,
not just the strengths.

Inner neurons hold their value from the previous timestep, so recurrence --
and therefore memory -- is something evolution can stumble into on its own.
"""

import math
import random

import capability_utils
import settings

# Endpoint kinds. A source is a sensor or an inner neuron; a sink is an inner
# neuron or an action.
SENSOR = 0
INNER = 1
ACTION = 2


class Gene(object):
    '''One connection in a brain.'''

    __slots__ = ('source_type', 'source_id', 'sink_type', 'sink_id', 'weight')

    def __init__(self, source_type, source_id, sink_type, sink_id, weight):
        self.source_type = source_type
        self.source_id = source_id
        self.sink_type = sink_type
        self.sink_id = sink_id
        self.weight = weight

    def copy(self):
        return Gene(self.source_type, self.source_id,
                    self.sink_type, self.sink_id, self.weight)

    def describe(self):
        '''Human-readable form, e.g. "border_distance --(-2.13)--> move_x".'''
        if self.source_type == SENSOR:
            source = capability_utils.SENSOR_NAMES[self.source_id]
        else:
            source = 'inner_{}'.format(self.source_id)

        if self.sink_type == INNER:
            sink = 'inner_{}'.format(self.sink_id)
        else:
            sink = capability_utils.ACTION_NAMES[self.sink_id]

        return '{} --({:+.2f})--> {}'.format(source, self.weight, sink)

    def __repr__(self):
        return '<Gene {}>'.format(self.describe())


# ----------------------------------------------------------------------------
# Building and mutating genomes
# ----------------------------------------------------------------------------

def random_gene():
    '''A single connection between two randomly chosen endpoints.'''
    source_type = random.choice((SENSOR, INNER))
    sink_type = random.choice((INNER, ACTION))
    return Gene(source_type=source_type,
                source_id=_random_source_id(source_type),
                sink_type=sink_type,
                sink_id=_random_sink_id(sink_type),
                weight=random.uniform(-settings.max_weight, settings.max_weight))


def random_genome(n_genes=None):
    '''A full genome of unbiased random connections, for generation zero.'''
    if n_genes is None:
        n_genes = settings.n_genes
    return [random_gene() for _ in range(n_genes)]


def mutate(genome, rate=None):
    '''
    Return a copy of the genome with point mutations applied.

    Each gene independently has `rate` chance of being altered. When a gene is
    hit, one of its five fields changes: rewiring an endpoint restructures the
    brain, while a weight change only retunes it.
    '''
    if rate is None:
        rate = settings.point_mutation_rate

    child = []
    for gene in genome:
        gene = gene.copy()
        if random.random() < rate:
            _mutate_gene_in_place(gene)
        child.append(gene)
    return child


def _mutate_gene_in_place(gene):
    field = random.choice(('source_type', 'source_id', 'sink_type', 'sink_id', 'weight'))

    if field == 'source_type':
        gene.source_type = SENSOR if gene.source_type == INNER else INNER
        # The new endpoint kind has a different id range, so redraw the id too.
        gene.source_id = _random_source_id(gene.source_type)
    elif field == 'source_id':
        gene.source_id = _random_source_id(gene.source_type)
    elif field == 'sink_type':
        gene.sink_type = ACTION if gene.sink_type == INNER else INNER
        gene.sink_id = _random_sink_id(gene.sink_type)
    elif field == 'sink_id':
        gene.sink_id = _random_sink_id(gene.sink_type)
    else:
        nudged = gene.weight + random.gauss(0, settings.weight_jitter)
        gene.weight = max(-settings.max_weight, min(settings.max_weight, nudged))


def _random_source_id(source_type):
    if source_type == SENSOR:
        return random.randrange(capability_utils.N_SENSORS)
    return random.randrange(settings.n_inner_neurons)


def _random_sink_id(sink_type):
    if sink_type == INNER:
        return random.randrange(settings.n_inner_neurons)
    return random.randrange(capability_utils.N_ACTIONS)


# ----------------------------------------------------------------------------
# The brain
# ----------------------------------------------------------------------------

class Brain(object):
    '''
    The runnable form of a genome.

    Construction sorts the genome's connections by what they feed, and works
    out which sensors this particular organism actually depends on. Most
    genomes ignore most senses, and skipping the unused ones is what keeps a
    generation cheap enough to watch in real time.
    '''

    def __init__(self, genome):
        self.genome = genome

        self.inner_connections = []   # (sink_id, source_type, source_id, weight)
        self.action_connections = []  # (sink_id, source_type, source_id, weight)
        needed = set()

        for gene in genome:
            connection = (gene.sink_id, gene.source_type, gene.source_id, gene.weight)
            if gene.sink_type == INNER:
                self.inner_connections.append(connection)
            else:
                self.action_connections.append(connection)
            if gene.source_type == SENSOR:
                needed.add(gene.source_id)

        self.needed_sensors = sorted(needed)
        self.inner_values = [0.0] * settings.n_inner_neurons

    def reset(self):
        '''Clear working memory. Called when an organism starts a generation.'''
        self.inner_values = [0.0] * settings.n_inner_neurons

    def think(self, org, world):
        '''
        Run one timestep of the brain and return a level in [-1, 1] per action.

        Inner neurons are updated from the sensors and from their own previous
        values; the actions are then driven by the sensors and the freshly
        updated inner neurons.
        '''
        sensors = {}
        for index in self.needed_sensors:
            sensors[index] = capability_utils.SENSORS[index][1](org, world)

        previous_inner = self.inner_values

        inner_sums = [0.0] * settings.n_inner_neurons
        for sink_id, source_type, source_id, weight in self.inner_connections:
            value = sensors[source_id] if source_type == SENSOR else previous_inner[source_id]
            inner_sums[sink_id] += value * weight
        new_inner = [math.tanh(total) for total in inner_sums]

        action_sums = [0.0] * capability_utils.N_ACTIONS
        for sink_id, source_type, source_id, weight in self.action_connections:
            value = sensors[source_id] if source_type == SENSOR else new_inner[source_id]
            action_sums[sink_id] += value * weight

        self.inner_values = new_inner
        return [math.tanh(total) for total in action_sums]

    def describe(self):
        '''The whole wiring diagram as text, one connection per line.'''
        return '\n'.join(gene.describe() for gene in self.genome)

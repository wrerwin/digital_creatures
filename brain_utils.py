"""
Genomes and the brains they build.

A genome is a fixed-length tuple of genes. Each gene describes one connection:

    source (a sensor, or an inner neuron)  --weight-->  sink (an inner neuron, or an action)

Because the *endpoints* are part of the gene rather than fixed in advance, a
mutation can rewire an organism onto a sense it never used before, or hand
control of an action to a different part of its brain. The topology evolves,
not just the strengths.

Inner neurons hold their value from the previous timestep, so recurrence --
and therefore memory -- is something evolution can stumble into on its own.

Genes are immutable, which is what makes reproduction safe: an offspring's
genome shares gene objects with its parent's and cannot disturb them.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import TYPE_CHECKING, Final

from capability_utils import SENSOR_FUNCTIONS, Action, Sensor

if TYPE_CHECKING:
    from organism import Organism, World
    from settings import Settings


class Source(IntEnum):
    """What drives a connection."""

    SENSOR = 0
    INNER = 1


class Sink(IntEnum):
    """What a connection drives."""

    INNER = 0
    ACTION = 1


@dataclass(frozen=True, slots=True)
class Gene:
    """One connection in a brain."""

    source_kind: Source
    source_id: int
    sink_kind: Sink
    sink_id: int
    weight: float

    def describe(self) -> str:
        """Human-readable form, e.g. `border_distance --(-2.13)--> move_x`."""
        source = (
            Sensor(self.source_id)
            if self.source_kind is Source.SENSOR
            else f"inner_{self.source_id}"
        )
        sink = f"inner_{self.sink_id}" if self.sink_kind is Sink.INNER else Action(self.sink_id)
        return f"{source} --({self.weight:+.2f})--> {sink}"


type Genome = tuple[Gene, ...]

_MUTABLE_FIELDS: Final = ("source_kind", "source_id", "sink_kind", "sink_id", "weight")


# ----------------------------------------------------------------------------
# Building and mutating genomes
# ----------------------------------------------------------------------------


def random_gene(config: Settings) -> Gene:
    """A single connection between two randomly chosen endpoints."""
    source_kind = random.choice(list(Source))
    sink_kind = random.choice(list(Sink))
    return Gene(
        source_kind=source_kind,
        source_id=_random_source_id(source_kind, config),
        sink_kind=sink_kind,
        sink_id=_random_sink_id(sink_kind, config),
        weight=random.uniform(-config.max_weight, config.max_weight),
    )


def random_genome(config: Settings) -> Genome:
    """A full genome of unbiased random connections, for generation zero."""
    return tuple(random_gene(config) for _ in range(config.n_genes))


def mutate(genome: Genome, config: Settings, rate: float | None = None) -> Genome:
    """
    Return a new genome with point mutations applied.

    Each gene independently has `rate` chance of being altered. When a gene is
    hit, one of its five fields changes: rewiring an endpoint restructures the
    brain, while a weight change only retunes it. Untouched genes are shared
    with the parent rather than copied, since genes are immutable.
    """
    if rate is None:
        rate = config.point_mutation_rate
    return tuple(_mutated(gene, config) if random.random() < rate else gene for gene in genome)


def _mutated(gene: Gene, config: Settings) -> Gene:
    """One gene with a single randomly chosen field changed."""
    match random.choice(_MUTABLE_FIELDS):
        case "source_kind":
            # The new endpoint kind has a different id range, so redraw the id too.
            kind = Source.SENSOR if gene.source_kind is Source.INNER else Source.INNER
            return replace(gene, source_kind=kind, source_id=_random_source_id(kind, config))
        case "source_id":
            return replace(gene, source_id=_random_source_id(gene.source_kind, config))
        case "sink_kind":
            kind = Sink.INNER if gene.sink_kind is Sink.ACTION else Sink.ACTION
            return replace(gene, sink_kind=kind, sink_id=_random_sink_id(kind, config))
        case "sink_id":
            return replace(gene, sink_id=_random_sink_id(gene.sink_kind, config))
        case _:
            nudged = gene.weight + random.gauss(0.0, config.weight_jitter)
            clamped = max(-config.max_weight, min(config.max_weight, nudged))
            return replace(gene, weight=clamped)


def _random_source_id(kind: Source, config: Settings) -> int:
    """Pick an endpoint from the capabilities this run allows, not from every one."""
    if kind is Source.SENSOR:
        return int(random.choice(config.enabled_sensors))
    return random.randrange(config.n_inner_neurons)


def _random_sink_id(kind: Sink, config: Settings) -> int:
    if kind is Sink.INNER:
        return random.randrange(config.n_inner_neurons)
    return int(random.choice(config.enabled_actions))


# ----------------------------------------------------------------------------
# The brain
# ----------------------------------------------------------------------------

type _Connection = tuple[int, bool, int, float]
"""(sink_id, whether the source is a sensor rather than an inner neuron, source_id, weight)"""


class Brain:
    """
    The runnable form of a genome.

    Construction sorts the genome's connections by what they feed and works out
    which sensors this particular organism actually depends on. Most genomes
    ignore most senses, and skipping the unused ones is what keeps a generation
    cheap enough to watch in real time.
    """

    __slots__ = ("_inner", "_to_actions", "_to_inner", "genome", "needed_sensors")

    def __init__(self, genome: Genome, config: Settings) -> None:
        self.genome = genome
        self._to_inner: list[_Connection] = []
        self._to_actions: list[_Connection] = []
        needed: set[Sensor] = set()

        for gene in genome:
            from_sensor = gene.source_kind is Source.SENSOR
            if from_sensor:
                needed.add(Sensor(gene.source_id))

            connection = (gene.sink_id, from_sensor, gene.source_id, gene.weight)
            target = self._to_inner if gene.sink_kind is Sink.INNER else self._to_actions
            target.append(connection)

        self.needed_sensors: tuple[Sensor, ...] = tuple(sorted(needed))
        self._inner = [0.0] * config.n_inner_neurons

    def reset(self) -> None:
        """Clear working memory. Called when an organism starts a generation."""
        self._inner = [0.0] * len(self._inner)

    def think(self, org: Organism, world: World) -> list[float]:
        """
        Run one timestep and return a level in [-1, 1] per action.

        Inner neurons update from the sensors and from their own previous
        values; the actions are then driven by the sensors and the freshly
        updated inner neurons.
        """
        readings = {
            int(sensor): SENSOR_FUNCTIONS[sensor](org, world) for sensor in self.needed_sensors
        }
        previous = self._inner

        inner_sums = [0.0] * len(previous)
        for sink_id, from_sensor, source_id, weight in self._to_inner:
            value = readings[source_id] if from_sensor else previous[source_id]
            inner_sums[sink_id] += value * weight
        current = [math.tanh(total) for total in inner_sums]

        action_sums = [0.0] * len(Action)
        for sink_id, from_sensor, source_id, weight in self._to_actions:
            value = readings[source_id] if from_sensor else current[source_id]
            action_sums[sink_id] += value * weight

        self._inner = current
        return [math.tanh(total) for total in action_sums]

    def describe(self) -> str:
        """The whole wiring diagram as text, one connection per line."""
        return "\n".join(gene.describe() for gene in self.genome)

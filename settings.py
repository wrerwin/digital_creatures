"""
Every tunable number in one place.

`Settings` is frozen and passed explicitly to the world that uses it, so a run
can be reconfigured without reaching in and rewriting module globals. Use
`dataclasses.replace` to derive a variant:

    from dataclasses import replace
    fast = replace(Settings(), steps_per_generation=50)
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    # World geometry. Positions are integer grid cells in [0, width) x [0, height).
    width: int = 100
    height: int = 100

    barrier_layout: str = "none"
    """Which obstacle layout to build into the world. See `barriers.LAYOUTS`."""

    # Population and timing
    n_organisms: int = 250
    steps_per_generation: int = 200
    n_generations: int = 100

    # Perception
    sense_radius: int = 6
    """How far an organism can feel neighbours and smell pheromone."""

    # The pheromone layer
    pheromone_decay: float = 0.92
    """Fraction of the pheromone on a cell that survives each timestep."""

    pheromone_deposit: float = 0.5
    """How much an organism lays down at full emission strength."""

    # Brain structure
    n_inner_neurons: int = 4
    """Internal neurons available to a genome as connection sources and sinks."""

    n_genes: int = 24
    """Connections per genome."""

    max_weight: float = 4.0
    """Gene weights are drawn from, and clamped to, [-max_weight, max_weight]."""

    # Reproduction
    point_mutation_rate: float = 0.02
    """Per-gene chance of a mutation when an offspring is produced."""

    weight_jitter: float = 0.4
    """Standard deviation of the nudge applied to a mutated weight."""

    # Selection
    survival_zone_fraction: float = 0.2
    """Fraction of the world's width that the zone objectives treat as safe."""

    stay_fraction: float = 0.5
    """For `stay`: the fraction of the generation that must be spent in the zone."""

    hazard_radius: float = 12.0
    """For `hazard`: radius of the roaming danger circle."""

    hazard_period: int = 100
    """For `hazard`: timesteps the hazard takes to complete one circuit."""


DEFAULT = Settings()

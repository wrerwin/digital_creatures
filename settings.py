"""
Every tunable number in one place.

`Settings` is frozen and passed explicitly to the world that uses it, so a run
can be reconfigured without reaching in and rewriting module globals. Use
`dataclasses.replace` to derive a variant:

    from dataclasses import replace
    fast = replace(Settings(), steps_per_generation=50)
"""

from collections.abc import Iterable
from dataclasses import dataclass, replace

from capability_utils import Action, Sensor


@dataclass(frozen=True, slots=True)
class Settings:
    # World geometry. Positions are integer grid cells in [0, width) x [0, height).
    width: int = 100
    height: int = 100

    # Which capabilities evolution is allowed to wire up. Switching one off
    # removes it from the search space entirely: no new gene will target it, so
    # whatever behaviour depended on it has to be found some other way -- or
    # cannot be found at all. Ordered tuples rather than sets, so that a run is
    # reproducible from a seed.
    enabled_sensors: tuple[Sensor, ...] = tuple(Sensor)
    enabled_actions: tuple[Action, ...] = tuple(Action)

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

    # --- Metabolism -------------------------------------------------------
    #
    # Energy is what stops more capability from being strictly better. A brain
    # pays upkeep for every sense it is wired to, whether or not that sense is
    # earning its keep, so a bloated genome starves before it reaches the zone.

    energy_enabled: bool = True
    """Whether organisms burn energy and can starve."""

    initial_energy: float = 140.0
    """
    Energy an organism starts each generation with.

    Set so a median random brain -- about nine distinct senses -- spends
    roughly 108 of it over a full generation and lives, while the bloated tail
    starves partway through. Lower it and generation zero is wiped out before
    selection has anything to work with.
    """

    metabolism: float = 0.10
    """Energy burned every timestep just by being alive."""

    sense_cost: float = 0.04
    """
    Energy burned per timestep for each distinct sense a brain consults.

    Tuned so the cost actually bites: at this rate a population sheds roughly
    two of its nine wired senses over thirty generations, while at half of it
    the pressure is lost in the noise.
    """

    move_cost: float = 0.08
    """Extra energy burned on a timestep the organism actually moves."""

    emit_cost: float = 0.30
    """Energy burned laying down pheromone at full strength."""

    # --- Reproduction -----------------------------------------------------

    reproduction: str = "asexual"
    """Which strategy fills the next generation. See `reproduction.STRATEGIES`."""

    point_mutation_rate: float = 0.02
    """Per-gene chance of a mutation when an offspring is produced."""

    weight_jitter: float = 0.4
    """Standard deviation of the nudge applied to a mutated weight."""

    offspring_per_survivor: float = 2.0
    """
    How many offspring each breeding survivor leaves.

    This is the knife's edge. At 2.0 the population holds steady at exactly 50%
    survival, grows above it and shrinks below -- so a population that slips
    even slightly is on its way out.
    """

    carrying_capacity: int = 400
    """Hard ceiling on population, however well the survivors do."""

    mating_radius: int = 18
    """
    For sexual reproduction: how far apart two survivors can be and still pair.

    A survivor with nobody in range does not breed at all, so sex adds a second
    problem on top of the objective -- reach the zone *and* not be alone.
    """

    # --- Selection --------------------------------------------------------

    survival_zone_fraction: float = 0.12
    """
    Fraction of the world's width that the zone objectives treat as safe.

    Deliberately mean. At 0.2 a population sails past replacement within a few
    generations and pins at the carrying capacity; at 0.12 it usually crashes
    first -- one measured run fell from 250 to 18 before recovering -- which is
    the knife's edge worth watching.
    """

    zone_shrink_per_generation: float = 0.0
    """
    Fraction the survival zone contracts by each generation.

    At 0 the target never moves. Above it, a solution that worked at generation
    10 stops working by generation 50, so the population has to keep adapting
    rather than settling.
    """

    min_zone_fraction: float = 0.03
    """Floor the shrinking zone will not contract past."""

    stay_fraction: float = 0.5
    """For `stay`: the fraction of the generation that must be spent in the zone."""

    hazard_radius: float = 12.0
    """For `hazard`: radius of the roaming danger circle."""

    hazard_period: int = 100
    """For `hazard`: timesteps the hazard takes to complete one circuit."""

    def __post_init__(self) -> None:
        # A population with nothing to perceive, or no way to act, cannot
        # evolve at all -- and would fail much later with a confusing
        # IndexError from deep inside gene creation.
        if not self.enabled_sensors:
            raise ValueError("at least one sensor must be enabled")
        if not self.enabled_actions:
            raise ValueError("at least one action must be enabled")
        if len(set(self.enabled_sensors)) != len(self.enabled_sensors):
            raise ValueError("enabled_sensors contains duplicates")
        if len(set(self.enabled_actions)) != len(self.enabled_actions):
            raise ValueError("enabled_actions contains duplicates")

    def with_capabilities(
        self,
        sensors: Iterable[Sensor] | None = None,
        actions: Iterable[Action] | None = None,
    ) -> "Settings":
        """
        A copy with a different set of capabilities available to evolution.

        Order is normalised so that two selections of the same capabilities
        produce identical settings, and therefore identical seeded runs.
        """
        return replace(
            self,
            enabled_sensors=tuple(sorted(sensors)) if sensors is not None else self.enabled_sensors,
            enabled_actions=tuple(sorted(actions)) if actions is not None else self.enabled_actions,
        )


DEFAULT = Settings()

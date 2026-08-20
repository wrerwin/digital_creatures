"""
How the survivors fill the next generation.

Two strategies, and the difference between them is not just genetics:

- **asexual** — a survivor is cloned with mutation. All new variation comes
  from mutation, and any survivor breeds regardless of where it ended up.

- **sexual** — survivors must *find each other*. Two survivors within
  `mating_radius` pair up and their genomes cross over; a survivor with nobody
  in range leaves no offspring at all. Reaching the zone is no longer enough,
  which is what makes the neighbour senses worth wiring.

Either way the population is no longer refilled to a fixed size. Each breeding
survivor leaves `offspring_per_survivor` young, capped by the carrying
capacity, so the population grows when the generation went well and shrinks
when it did not -- and can reach zero, which ends the run.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Final

import brain_utils

if TYPE_CHECKING:
    from brain_utils import Genome
    from organism import Organism
    from settings import Settings


class Strategy(ABC):
    """A way of turning this generation's survivors into the next generation."""

    name: str = "strategy"

    needs_partners: bool = False
    """Whether a lone survivor is unable to breed."""

    @abstractmethod
    def breeding_pairs(self, survivors: list[Organism], config: Settings) -> list[Parents]:
        """Who actually gets to breed, as a list of one or two parents."""

    def offspring(self, parents: Parents, config: Settings) -> Organism:
        """One child from a set of parents."""
        from organism import Organism  # imported late: organism imports this module

        genome = self.combine(parents, config)
        child = Organism(config, genome=brain_utils.mutate(genome, config))
        # A child belongs to one parent's line. With two parents that is a
        # coin toss, which is what lets a lineage die out even while its genes
        # live on in somebody else's descendants.
        child.lineage = random.choice(parents).lineage
        return child

    @abstractmethod
    def combine(self, parents: Parents, config: Settings) -> Genome:
        """The genome a child inherits before mutation."""


type Parents = list["Organism"]


class Asexual(Strategy):
    """Clone a survivor. Mutation is the only source of new variation."""

    name = "asexual"

    def breeding_pairs(self, survivors: list[Organism], config: Settings) -> list[Parents]:
        return [[survivor] for survivor in survivors]

    def combine(self, parents: Parents, config: Settings) -> Genome:
        return parents[0].genome


class Sexual(Strategy):
    """
    Pair survivors that ended the generation near each other, and cross their genomes.

    Pairing is greedy and monogamous: each survivor takes the nearest unpaired
    partner within range. Anyone left over does not breed, so a population that
    scatters can meet the objective and still collapse.
    """

    name = "sexual"
    needs_partners = True

    def breeding_pairs(self, survivors: list[Organism], config: Settings) -> list[Parents]:
        unpaired = list(survivors)
        random.shuffle(unpaired)  # no positional advantage in who pairs first
        taken: set[int] = set()
        pairs: list[Parents] = []

        for index, candidate in enumerate(unpaired):
            if index in taken:
                continue
            partner = self._nearest_free(unpaired, index, taken, config.mating_radius)
            if partner is None:
                continue
            taken.add(index)
            taken.add(partner)
            pairs.append([candidate, unpaired[partner]])

        return pairs

    @staticmethod
    def _nearest_free(
        survivors: list[Organism], index: int, taken: set[int], radius: int
    ) -> int | None:
        """Index of the closest unpaired survivor within range, if there is one."""
        here = survivors[index]
        best: int | None = None
        best_distance = radius + 1

        for other_index, other in enumerate(survivors):
            if other_index == index or other_index in taken:
                continue
            distance = max(abs(other.x - here.x), abs(other.y - here.y))
            if distance <= radius and distance < best_distance:
                best, best_distance = other_index, distance

        return best

    def combine(self, parents: Parents, config: Settings) -> Genome:
        """
        Uniform crossover: every connection comes from one parent or the other.

        Genes here are independent connections rather than a linked sequence,
        so there is nothing for a single crossover point to preserve -- picking
        per gene mixes the parents far more thoroughly.
        """
        mother, father = parents[0], parents[1]
        return tuple(
            random.choice((from_mother, from_father))
            for from_mother, from_father in zip(mother.genome, father.genome, strict=True)
        )


STRATEGIES: Final[dict[str, Strategy]] = {
    strategy.name: strategy for strategy in (Asexual(), Sexual())
}


def resolve(strategy: Strategy | str | None) -> Strategy:
    """Accept a strategy, the name of one, or nothing at all."""
    if strategy is None:
        return STRATEGIES["asexual"]
    if isinstance(strategy, str):
        try:
            return STRATEGIES[strategy]
        except KeyError:
            known = ", ".join(STRATEGIES)
            raise ValueError(
                f"unknown reproduction strategy {strategy!r}; try one of: {known}"
            ) from None
    return strategy


def next_generation(
    survivors: list[Organism], strategy: Strategy, config: Settings
) -> list[Organism]:
    """
    Build the next generation, or an empty list if the population dies out.

    The size is earned rather than fixed: breeding survivors leave
    `offspring_per_survivor` young each, capped by the carrying capacity. The
    fractional part is settled by a coin toss so that a rate of 2.5 really does
    average two and a half rather than quietly rounding down every time.
    """
    if not survivors:
        return []

    pairs = strategy.breeding_pairs(survivors, config)
    if not pairs:
        return []

    breeding = sum(len(pair) for pair in pairs)
    exact = breeding * config.offspring_per_survivor
    total = int(exact)
    if random.random() < exact - total:
        total += 1
    total = min(total, config.carrying_capacity)

    return [strategy.offspring(random.choice(pairs), config) for _ in range(total)]

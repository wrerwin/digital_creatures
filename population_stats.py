"""
What the population as a whole looks like, rather than one creature at a time.

Reading a single brain tells you what one creature does. Reading the whole
population tells you what evolution has *decided* -- which senses it kept, which
it abandoned, how much of the gene pool one founding line has taken over.

`expression` is the summary the UI charts and the terminal prints. It is
deliberately cheap: one pass over the genomes, once per generation.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from brain_utils import Sink, Source
from capability_utils import Action, Sensor

if TYPE_CHECKING:
    from organism import Organism, World


def expression(world: World) -> dict[str, Any]:
    """
    A snapshot of how the population's genes are being expressed.

    For each capability: what share of the population wires it at all, how many
    connections the average creature devotes to it, and how strong those
    connections are. Share is the interesting one -- a sense that drops to zero
    has been actively selected away, and one that climbs to 1.0 has become
    load-bearing.
    """
    organisms = world.organisms
    if not organisms:
        return _empty()

    total = len(organisms)
    sensor_users: Counter[Sensor] = Counter()
    sensor_links: Counter[Sensor] = Counter()
    sensor_weight: Counter[Sensor] = Counter()
    action_users: Counter[Action] = Counter()
    action_links: Counter[Action] = Counter()
    action_weight: Counter[Action] = Counter()

    senses_used = 0
    for org in organisms:
        seen_sensors: set[Sensor] = set()
        seen_actions: set[Action] = set()
        for gene in org.genome:
            if gene.source_kind is Source.SENSOR:
                sensor = Sensor(gene.source_id)
                seen_sensors.add(sensor)
                sensor_links[sensor] += 1
                sensor_weight[sensor] += abs(gene.weight)
            if gene.sink_kind is Sink.ACTION:
                action = Action(gene.sink_id)
                seen_actions.add(action)
                action_links[action] += 1
                action_weight[action] += abs(gene.weight)

        sensor_users.update(seen_sensors)
        action_users.update(seen_actions)
        senses_used += len(seen_sensors)

    return {
        "population": total,
        "sensors": [
            _entry(
                str(sensor),
                sensor_users[sensor],
                sensor_links[sensor],
                sensor_weight[sensor],
                total,
            )
            for sensor in Sensor
        ],
        "actions": [
            _entry(
                str(action),
                action_users[action],
                action_links[action],
                action_weight[action],
                total,
            )
            for action in Action
        ],
        "mean_senses_used": senses_used / total,
        "mean_upkeep": sum(org.upkeep for org in organisms) / total,
        "lineages": lineages(world),
        "energy": energy_spread(world),
    }


def _entry(label: str, users: int, links: int, weight: float, total: int) -> dict[str, Any]:
    return {
        "label": label,
        "share": users / total,
        "links_per_organism": links / total,
        "mean_weight": weight / links if links else 0.0,
    }


def lineages(world: World) -> dict[str, Any]:
    """
    How much of the population each founding line still accounts for.

    A run that collapses to one lineage has lost its variation, and will only
    find anything new by mutation from then on.
    """
    founding = world.founding_lineages
    counts = Counter(org.lineage for org in world.organisms)
    if not counts:
        return {
            "alive": 0,
            "founding": founding,
            "remaining": 0.0,
            "dominant_share": 0.0,
            "top": [],
        }

    total = sum(counts.values())
    return {
        "alive": len(counts),
        "founding": founding,
        "remaining": len(counts) / founding if founding else 0.0,
        "dominant_share": counts.most_common(1)[0][1] / total,
        "top": [{"lineage": line, "share": count / total} for line, count in counts.most_common(8)],
    }


def energy_spread(world: World) -> dict[str, float]:
    """Where the population's energy ended up, as a rough sense of how close it ran."""
    living = [org.energy for org in world.organisms if org.alive]
    if not living:
        return {"mean": 0.0, "lowest": 0.0, "highest": 0.0}
    return {
        "mean": sum(living) / len(living),
        "lowest": min(living),
        "highest": max(living),
    }


def _empty() -> dict[str, Any]:
    return {
        "population": 0,
        "sensors": [_entry(str(s), 0, 0, 0.0, 1) for s in Sensor],
        "actions": [_entry(str(a), 0, 0, 0.0, 1) for a in Action],
        "mean_senses_used": 0.0,
        "mean_upkeep": 0.0,
        "lineages": {
            "alive": 0,
            "founding": 0,
            "remaining": 0.0,
            "dominant_share": 0.0,
            "top": [],
        },
        "energy": {"mean": 0.0, "lowest": 0.0, "highest": 0.0},
    }


def summarise(world: World, width: int = 34) -> str:
    """
    The same picture as text, for the terminal runner.

    Only the capabilities anybody still uses are listed: a population that has
    abandoned two thirds of its senses should look like it has.
    """
    stats = expression(world)
    lineage = stats["lineages"]
    lines = [
        f"population {stats['population']}   "
        f"lineages {lineage['alive']} of {lineage['founding']} "
        f"({lineage['remaining']:.0%} remaining)   "
        f"mean senses wired {stats['mean_senses_used']:.1f}"
    ]

    for kind in ("sensors", "actions"):
        used = [item for item in stats[kind] if item["share"] > 0.0]
        if not used:
            continue
        lines.append(f"\n{kind}:")
        for item in sorted(used, key=lambda entry: -entry["share"]):
            filled = round(item["share"] * width)
            bar = "#" * filled + "." * (width - filled)
            lines.append(f"  {item['label']:>20} {bar} {item['share']:5.0%}")

    return "\n".join(lines)


def organism_row(org: Organism) -> dict[str, Any]:
    """One creature's headline numbers, for inspection."""
    return {
        "lineage": org.lineage,
        "alive": org.alive,
        "energy": org.energy,
        "upkeep": org.upkeep,
        "senses_wired": len(org.brain.needed_sensors),
    }

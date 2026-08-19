"""
Tools for keeping, reloading and looking at evolved creatures.

A run that produces something interesting is worth nothing if the genome
disappears when the process exits. Genomes are saved as JSON keyed by *name*
rather than by enum value, so a file stays readable after new senses or actions
are added to `capability_utils`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from brain_utils import Gene, Sink, Source
from capability_utils import Action, Sensor

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from brain_utils import Brain, Genome
    from settings import Settings

FORMAT_VERSION = 1


# ----------------------------------------------------------------------------
# Saving and loading
# ----------------------------------------------------------------------------


def genome_to_dict(genome: Genome, config: Settings) -> dict[str, Any]:
    """A JSON-ready description of a genome, using names rather than indices."""
    return {
        "version": FORMAT_VERSION,
        "n_inner_neurons": config.n_inner_neurons,
        "genes": [
            {
                "source": _endpoint_name(gene.source_kind, gene.source_id),
                "sink": _endpoint_name(gene.sink_kind, gene.sink_id),
                # Written at full precision: a genome that shifts on reload is
                # a different creature from the one that was worth keeping.
                "weight": gene.weight,
            }
            for gene in genome
        ],
    }


def genome_from_dict(data: dict[str, Any]) -> Genome:
    """Rebuild a genome saved by `genome_to_dict`."""
    version = data.get("version")
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported genome format {version!r}, expected {FORMAT_VERSION}")

    genes = []
    for entry in data["genes"]:
        source_kind, source_id = _parse_endpoint(entry["source"], as_source=True)
        sink_kind, sink_id = _parse_endpoint(entry["sink"], as_source=False)
        genes.append(Gene(source_kind, source_id, sink_kind, sink_id, float(entry["weight"])))
    return tuple(genes)


def save_genome(genome: Genome, config: Settings, path: str | Path) -> Path:
    """Write one genome to a JSON file."""
    path = Path(path)
    path.write_text(json.dumps(genome_to_dict(genome, config), indent=2), encoding="utf-8")
    return path


def load_genome(path: str | Path) -> Genome:
    """Read a genome written by `save_genome`."""
    return genome_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _endpoint_name(kind: Source | Sink, index: int) -> str:
    if kind is Source.SENSOR:
        return str(Sensor(index))
    if kind is Sink.ACTION:
        return str(Action(index))
    return f"inner_{index}"


def _parse_endpoint(name: str, as_source: bool) -> tuple[Any, int]:
    """Turn a saved endpoint name back into its (kind, id) pair."""
    if name.startswith("inner_"):
        kind = Source.INNER if as_source else Sink.INNER
        return kind, int(name.removeprefix("inner_"))

    if as_source:
        return Source.SENSOR, int(Sensor[name.upper()])
    return Sink.ACTION, int(Action[name.upper()])


# ----------------------------------------------------------------------------
# Drawing a brain
# ----------------------------------------------------------------------------


def draw_brain(brain: Brain, config: Settings, axes: Axes | None = None) -> Axes:
    """
    Draw a brain as a wiring diagram: senses left, inner neurons middle, actions right.

    Line thickness follows the strength of a connection and colour follows its
    sign, so the shape of a behaviour is visible at a glance in a way the text
    listing never manages. Only the senses and inner neurons the genome
    actually uses are drawn.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(figsize=(9, 7))

    used_inner = sorted(
        {gene.source_id for gene in brain.genome if gene.source_kind is Source.INNER}
        | {gene.sink_id for gene in brain.genome if gene.sink_kind is Sink.INNER}
    )
    used_actions = sorted({gene.sink_id for gene in brain.genome if gene.sink_kind is Sink.ACTION})

    sensor_pos = _column([str(s) for s in brain.needed_sensors], x=0.0)
    inner_pos = _column([f"inner_{i}" for i in used_inner], x=1.0)
    action_pos = _column([str(Action(a)) for a in used_actions], x=2.0)

    def locate(kind: Source | Sink, index: int, source: bool) -> tuple[float, float] | None:
        if source and kind is Source.SENSOR:
            return sensor_pos.get(str(Sensor(index)))
        if not source and kind is Sink.ACTION:
            return action_pos.get(str(Action(index)))
        return inner_pos.get(f"inner_{index}")

    strongest = max((abs(gene.weight) for gene in brain.genome), default=1.0) or 1.0

    for gene in brain.genome:
        start = locate(gene.source_kind, gene.source_id, source=True)
        end = locate(gene.sink_kind, gene.sink_id, source=False)
        if start is None or end is None:
            continue

        colour = "#2c7fb8" if gene.weight >= 0 else "#d95f0e"
        width = 0.4 + 3.0 * abs(gene.weight) / strongest

        if start == end:  # a self-loop: an inner neuron feeding itself
            axes.annotate(
                "",
                xy=(start[0] + 0.08, start[1] + 0.02),
                xytext=(start[0] + 0.08, start[1] - 0.02),
                arrowprops={
                    "arrowstyle": "->",
                    "color": colour,
                    "lw": width,
                    "connectionstyle": "arc3,rad=3.0",
                },
            )
            continue

        axes.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "->",
                "color": colour,
                "lw": width,
                "alpha": 0.75,
                "shrinkA": 6,
                "shrinkB": 6,
                "connectionstyle": "arc3,rad=0.12",
            },
        )

    for positions, align, style in (
        (sensor_pos, "right", {"color": "#31a354"}),
        (inner_pos, "center", {"color": "#756bb1"}),
        (action_pos, "left", {"color": "#e6550d"}),
    ):
        for label, (x, y) in positions.items():
            axes.text(
                x,
                y,
                label,
                ha=align,
                va="center",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": style["color"]},
            )

    axes.set_xlim(-0.6, 2.6)
    axes.set_ylim(-0.1, 1.1)
    axes.axis("off")
    axes.set_title("senses  →  inner neurons  →  actions")
    return axes


def _column(labels: list[str], x: float) -> dict[str, tuple[float, float]]:
    """Evenly space labels down a vertical column."""
    if not labels:
        return {}
    if len(labels) == 1:
        return {labels[0]: (x, 0.5)}
    gap = 1.0 / (len(labels) - 1)
    return {label: (x, i * gap) for i, label in enumerate(labels)}

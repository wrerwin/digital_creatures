"""
Run the simulation.

    uv run python execute.py                              # reach the west wall
    uv run python execute.py --objective stay             # and hold position there
    uv run python execute.py --barriers slalom            # make it work for it
    uv run python execute.py --compare left,stay,corners  # race objectives, headless
    uv run python execute.py --watch 0                    # no animation, just numbers

Generation 0 is random noise. Watch the survival percentage climb.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import barriers
import inspect_utils
import population_stats
import reproduction
from objectives import OBJECTIVES
from organism import Organism, World
from settings import Settings

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    defaults = Settings()
    parser.add_argument(
        "--objective",
        choices=list(OBJECTIVES),
        default="left",
        help="what a creature has to do to earn the right to reproduce",
    )
    parser.add_argument(
        "--barriers",
        choices=sorted(barriers.LAYOUTS),
        default=defaults.barrier_layout,
        help="obstacle layout built into the world",
    )
    parser.add_argument(
        "--reproduction",
        choices=list(reproduction.STRATEGIES),
        default=defaults.reproduction,
        help="asexual clones survivors; sexual pairs nearby survivors and crosses their genomes",
    )
    parser.add_argument(
        "--no-metabolism",
        action="store_true",
        help="switch off energy, so brains cost nothing to run and nobody starves",
    )
    parser.add_argument(
        "--zone-shrink",
        type=float,
        default=defaults.zone_shrink_per_generation,
        metavar="F",
        help="fraction the survival zone contracts by each generation",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="print how the population's genes are being expressed at the end",
    )
    parser.add_argument("--generations", type=int, default=defaults.n_generations)
    parser.add_argument("--population", type=int, default=defaults.n_organisms)
    parser.add_argument(
        "--steps",
        type=int,
        default=defaults.steps_per_generation,
        help="timesteps per generation",
    )
    parser.add_argument(
        "--watch",
        type=int,
        default=10,
        metavar="N",
        help="animate every Nth generation (0 to disable animation)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed the random number generators for a repeatable run",
    )
    parser.add_argument(
        "--compare",
        metavar="A,B,C",
        default=None,
        help="run several objectives headless and plot their survival curves together",
    )
    parser.add_argument(
        "--save-genome",
        metavar="PATH",
        default=None,
        help="write a final creature's genome to a JSON file",
    )
    parser.add_argument(
        "--load-genome",
        metavar="PATH",
        default=None,
        help="seed the whole starting population from a saved genome",
    )
    parser.add_argument(
        "--draw-brain",
        action="store_true",
        help="show a wiring diagram of one final creature",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Settings:
    return replace(
        Settings(),
        n_organisms=args.population,
        steps_per_generation=args.steps,
        n_generations=args.generations,
        barrier_layout=args.barriers,
        reproduction=args.reproduction,
        energy_enabled=not args.no_metabolism,
        zone_shrink_per_generation=args.zone_shrink,
    )


class Animator:
    """Live view of the population, its obstacles, its scent trails and its goal."""

    def __init__(self, world: World, title: str) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        self.title = title

        self.figure: Figure
        self.axes: Axes
        self.figure, self.axes = plt.subplots(figsize=(6.5, 6.5))
        extent = (0, world.width, 0, world.height)

        # imshow indexes [row, column] = [y, x], so every mask is transposed.
        self.zone_images = []
        for shading in world.objective.zones(world):
            self.zone_images.append(
                self.axes.imshow(
                    shading.mask.T,
                    origin="lower",
                    cmap=shading.colour,
                    alpha=0.3,
                    extent=extent,
                    vmin=0,
                    vmax=1,
                    zorder=0,
                )
            )

        self.pheromone_image = self.axes.imshow(
            world.pheromone.T,
            origin="lower",
            cmap="BuPu",
            alpha=0.55,
            extent=extent,
            vmin=0,
            vmax=1,
            zorder=1,
        )
        # Solid cells are drawn opaque, empty ones fully transparent.
        self.axes.imshow(
            np.ma.masked_where(~world.barriers.T, world.barriers.T),
            origin="lower",
            cmap="Greys",
            alpha=0.85,
            extent=extent,
            vmin=0,
            vmax=1,
            zorder=2,
        )

        self.scatter = self.axes.scatter([], [], s=7, zorder=3)
        self.axes.set_xlim(0, world.width)
        self.axes.set_ylim(0, world.height)
        self.dynamic_zones = world.objective.dynamic

        plt.ion()
        plt.show(block=False)

    def draw(self, world: World, step: int) -> None:
        xs, ys = world.positions()
        self.scatter.set_offsets(np.column_stack((xs, ys)) if len(xs) else np.empty((0, 2)))
        self.pheromone_image.set_data(np.clip(world.pheromone.T, 0, 1))

        if self.dynamic_zones:
            shadings = world.objective.zones(world)
            for image, shading in zip(self.zone_images, shadings, strict=False):
                image.set_data(shading.mask.T)

        self.axes.set_title(
            f"{self.title} | generation {world.generation} | step {step} | alive {len(xs)}"
        )
        self.figure.canvas.draw_idle()
        self.plt.pause(0.001)

    def close(self) -> None:
        self.plt.ioff()
        self.plt.close(self.figure)


def seed_population(world: World, path: str) -> None:
    """Replace every starting genome with one loaded from disk."""
    genome = inspect_utils.load_genome(path)
    world.organisms = [Organism(world.config, genome=genome) for _ in range(world.n_organisms)]
    world.found_lineages(world.organisms)
    world.reset_grid(world.organisms)
    print(f"seeded {world.n_organisms} organisms from {path}")


def run(args: argparse.Namespace) -> World:
    """Run one objective, animating as configured, and return the finished world."""
    config = build_config(args)
    world = World(config=config, objective=args.objective)

    if args.load_genome:
        seed_population(world, args.load_genome)

    title = args.objective if args.barriers == "none" else f"{args.objective} / {args.barriers}"
    animator = Animator(world, title) if args.watch else None

    metabolism = "metabolism off" if args.no_metabolism else "metabolism on"
    print(
        f"{config.n_organisms} organisms, {config.steps_per_generation} steps per generation, "
        f'"{args.objective}" objective, "{args.barriers}" barriers, '
        f"{args.reproduction} reproduction, {metabolism}"
    )

    started = time.monotonic()
    try:
        for generation in range(config.n_generations):
            animating = animator is not None and generation % args.watch == 0
            before = world.population
            survivors = world.run_generation(on_step=animator.draw if animating else None)

            share = survivors / max(before, 1)
            elapsed = time.monotonic() - started
            print(
                f"generation {generation:>4}   "
                f"survivors {survivors:>4} / {before:<4} ({share:6.1%})   "
                f"next pop {world.population:>4}   {elapsed:6.1f}s"
            )
            if world.extinct:
                print("\nextinct: nobody left to breed.")
                break
    except KeyboardInterrupt:
        print("\nstopped early")
    finally:
        if animator is not None:
            animator.close()

    return world


def compare(args: argparse.Namespace) -> None:
    """Race several objectives against each other and plot the result."""
    import matplotlib.pyplot as plt

    names = [name.strip() for name in args.compare.split(",") if name.strip()]
    unknown = [name for name in names if name not in OBJECTIVES]
    if unknown:
        raise SystemExit(f"unknown objective(s): {', '.join(unknown)}")

    config = build_config(args)
    _, axes = plt.subplots(figsize=(8, 5))

    for name in names:
        world = World(config=config, objective=name)
        started = time.monotonic()
        curve = [world.run_generation() / config.n_organisms for _ in range(config.n_generations)]
        axes.plot(curve, label=name)
        print(f"{name:>18}   final {curve[-1]:6.1%}   {time.monotonic() - started:6.1f}s")

    axes.set_xlabel("generation")
    axes.set_ylabel("fraction surviving")
    axes.set_ylim(0, 1)
    axes.set_title(f"objectives compared ({args.barriers} barriers)")
    axes.legend()
    plt.show()


def report(world: World, args: argparse.Namespace) -> None:
    """Print, save and optionally draw one creature from the final generation."""
    if not world.organisms:
        return

    if args.stats:
        print("\nhow the population's genes are being expressed:")
        print(population_stats.summarise(world))

    example = world.organisms[0]
    consulted = ", ".join(str(s) for s in example.brain.needed_sensors) or "none"
    print("\nwiring of one organism from the final generation:")
    print(example.brain.describe())
    print(f"\nsenses it actually consults: {consulted}")

    if args.save_genome:
        path = inspect_utils.save_genome(example.genome, world.config, Path(args.save_genome))
        print(f"genome written to {path}")

    if args.draw_brain:
        import matplotlib.pyplot as plt

        inspect_utils.draw_brain(example.brain, world.config)
        plt.show()


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.compare:
        compare(args)
        return

    world = run(args)
    report(world, args)


if __name__ == "__main__":
    main()

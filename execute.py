"""
Run the simulation.

    uv run python execute.py                      # survive by reaching the west wall
    uv run python execute.py --criterion corners  # a different selection pressure
    uv run python execute.py --watch 0            # no animation, just the numbers

Generation 0 is random noise. Watch the survival percentage climb.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from organism import CRITERIA, World
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
        "--criterion",
        choices=sorted(CRITERIA),
        default="left",
        help="which survival criterion selects who reproduces",
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
    return parser.parse_args()


class Animator:
    """Live scatter plot of the population, with the survival zone shaded."""

    def __init__(self, world: World, criterion_name: str) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        self.criterion_name = criterion_name

        self.figure: Figure
        self.axes: Axes
        self.figure, self.axes = plt.subplots(figsize=(6, 6))

        # imshow indexes [row, column] = [y, x], so transpose the [x, y] mask.
        self.axes.imshow(
            world.survival_zone_mask().T,
            origin="lower",
            cmap="Greens",
            alpha=0.25,
            extent=(0, world.width, 0, world.height),
        )
        self.scatter = self.axes.scatter([], [], s=6)
        self.axes.set_xlim(0, world.width)
        self.axes.set_ylim(0, world.height)

        plt.ion()
        plt.show(block=False)

    def draw(self, world: World, step: int) -> None:
        xs, ys = world.positions()
        self.scatter.set_offsets(np.column_stack((xs, ys)))
        self.axes.set_title(
            f"{self.criterion_name} criterion | generation {world.generation} | step {step}"
        )
        self.figure.canvas.draw_idle()
        self.plt.pause(0.001)

    def close(self) -> None:
        self.plt.ioff()
        self.plt.close(self.figure)


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    config = replace(
        Settings(),
        n_organisms=args.population,
        steps_per_generation=args.steps,
        n_generations=args.generations,
    )
    world = World(config=config, criterion=CRITERIA[args.criterion])

    animator = Animator(world, args.criterion) if args.watch else None

    print(
        f"{config.n_organisms} organisms, {config.steps_per_generation} steps "
        f'per generation, "{args.criterion}" survival criterion'
    )

    started = time.monotonic()
    try:
        for generation in range(config.n_generations):
            animating = animator is not None and generation % args.watch == 0
            survivors = world.run_generation(on_step=animator.draw if animating else None)

            share = survivors / config.n_organisms
            elapsed = time.monotonic() - started
            print(
                f"generation {generation:>4}   "
                f"survivors {survivors:>4} / {config.n_organisms:<4} ({share:6.1%})   "
                f"{elapsed:6.1f}s"
            )
    except KeyboardInterrupt:
        print("\nstopped early")
    finally:
        if animator is not None:
            animator.close()

    show_example_brain(world)


def show_example_brain(world: World) -> None:
    """Print one organism's wiring, so the evolved behaviour can be read back."""
    if not world.organisms:
        return
    example = world.organisms[0]
    consulted = ", ".join(str(s) for s in example.brain.needed_sensors) or "none"
    print("\nwiring of one organism from the final generation:")
    print(example.brain.describe())
    print(f"\nsenses it actually consults: {consulted}")


if __name__ == "__main__":
    main()

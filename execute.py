"""
Run the simulation.

    python execute.py                      # default: survive by reaching the west wall
    python execute.py --criterion corners  # try a different selection pressure
    python execute.py --watch 0            # no animation, just the survival numbers

Generation 0 is random noise. Watch the survival percentage climb.
"""

import argparse
import random
import time

import numpy as np

import capability_utils
import settings
from organism import CRITERIA, world


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--criterion', choices=sorted(CRITERIA), default='left',
                        help='which survival criterion selects who reproduces')
    parser.add_argument('--generations', type=int, default=settings.n_generations)
    parser.add_argument('--population', type=int, default=settings.n_organisms)
    parser.add_argument('--steps', type=int, default=settings.steps_per_generation,
                        help='timesteps per generation')
    parser.add_argument('--watch', type=int, default=10, metavar='N',
                        help='animate every Nth generation (0 to disable animation)')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed the random number generators for a repeatable run')
    return parser.parse_args()


def survival_zone_mask(criterion, sim):
    '''
    A boolean grid of the cells that count as survivable.

    Rather than hard-coding a shape per criterion, this asks the criterion
    itself about every cell, so any new criterion draws itself correctly.
    '''
    probe = _Probe()
    mask = np.zeros((sim.width, sim.height), dtype=bool)
    for x in range(sim.width):
        for y in range(sim.height):
            probe.x = x
            probe.y = y
            mask[x, y] = criterion(probe, sim)
    return mask


class _Probe(object):
    '''Stand-in organism used only to ask a criterion about a bare position.'''
    __slots__ = ('x', 'y')


class Animator(object):
    '''Live scatter plot of the population, with the survival zone shaded.'''

    def __init__(self, sim, criterion, criterion_name):
        import matplotlib.pyplot as plt
        self.plt = plt

        self.figure, self.axes = plt.subplots(figsize=(6, 6))
        mask = survival_zone_mask(criterion, sim)
        # imshow indexes [row, column] = [y, x], so transpose the [x, y] mask.
        self.axes.imshow(mask.T, origin='lower', cmap='Greens', alpha=0.25,
                         extent=(0, sim.width, 0, sim.height))
        self.scatter = self.axes.scatter([], [], s=6)
        self.axes.set_xlim(0, sim.width)
        self.axes.set_ylim(0, sim.height)
        self.criterion_name = criterion_name

        plt.ion()
        plt.show(block=False)

    def draw(self, sim, step):
        xs, ys = sim.positions()
        self.scatter.set_offsets(np.column_stack((xs, ys)))
        self.axes.set_title('{} criterion | generation {} | step {}'.format(
            self.criterion_name, sim.generation, step))
        self.figure.canvas.draw_idle()
        self.plt.pause(0.001)

    def close(self):
        self.plt.ioff()
        self.plt.close(self.figure)


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    settings.steps_per_generation = args.steps
    settings.n_organisms = args.population

    criterion = CRITERIA[args.criterion]
    sim = world(n_organisms=args.population, survival_criterion=criterion)

    animator = None
    if args.watch:
        animator = Animator(sim, criterion, args.criterion)

    print('{} organisms, {} steps per generation, "{}" survival criterion'.format(
        args.population, args.steps, args.criterion))

    started = time.time()
    try:
        for generation in range(args.generations):
            animating = animator is not None and generation % args.watch == 0
            on_step = animator.draw if animating else None

            survivors = sim.run_generation(on_step=on_step)

            print('generation {:>4}   survivors {:>4} / {:<4} ({:5.1f}%)   {:6.1f}s'.format(
                generation, survivors, args.population,
                100.0 * survivors / args.population,
                time.time() - started))
    except KeyboardInterrupt:
        print('\nstopped early')
    finally:
        if animator is not None:
            animator.close()

    show_example_brain(sim)


def show_example_brain(sim):
    '''Print one organism's wiring, so the evolved behaviour can be read back.'''
    if not sim.organism_states:
        return
    example = sim.organism_states[0]
    print('\nwiring of one organism from the final generation:')
    print(example.brain.describe())
    print('\nsenses it actually consults: {}'.format(
        ', '.join(capability_utils.SENSOR_NAMES[i]
                  for i in example.brain.needed_sensors) or 'none'))


if __name__ == '__main__':
    main()

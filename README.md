# digital creatures

Little guys on a grid. Nobody tells them what to do — the only rule is who gets
to reproduce, and the behaviour that follows from that is the interesting part.

```bash
python execute.py
```

Generation 0 is random noise; watch the survival percentage climb.

## How it works

A generation runs for a fixed number of timesteps. At the end, a **survival
criterion** decides who reproduces — by default, "be in the left fifth of the
world". Survivors are cloned with mutation until the population is full again,
the grid is cleared, and the next generation starts.

Each organism has a **genome**: a fixed-length list of genes, where every gene
is one connection.

```
source (a sensor, or an inner neuron)  --weight-->  sink (an inner neuron, or an action)
```

Because the endpoints are encoded in the gene rather than fixed in advance, the
*topology* evolves and not just the weights. A mutation can rewire a creature
onto a sense it never used, or hand an action over to a different part of its
brain. Inner neurons keep their value between timesteps, so recurrence — and
therefore memory — is something evolution can find on its own.

Every organism's wiring can be read back as text, which is how you work out
what a behaviour actually *is*:

```
border_distance --(-2.13)--> move_x
inner_2 --(+1.44)--> move_forward
```

## Files

| file | what's in it |
| --- | --- |
| `settings.py` | every tunable number: world size, population, mutation rate, genome length |
| `capability_utils.py` | what a creature can sense and what it can do — the two lists that define its interface to the world |
| `brain_utils.py` | genes, mutation, and the network a genome builds |
| `organism.py` | organisms, the grid world, the generational cycle, and the survival criteria |
| `execute.py` | the runner and the live animation |
| `tests.py` | invariant checks — run `python tests.py` |
| `sandbox.ipynb` | design notes and a scratchpad for poking at individual creatures |

## Knobs worth turning

```bash
python execute.py --criterion corners   # left, right, centre, corners
python execute.py --watch 0             # skip the animation, just print numbers
python execute.py --watch 5             # animate every 5th generation
python execute.py --seed 42             # repeatable run
```

The survival criterion is the whole selection pressure. Adding one is a
four-line function at the bottom of `organism.py` plus an entry in `CRITERIA`;
the animation works out how to shade the new zone by itself.

Adding a sense or an action is a single entry in `capability_utils.py` — no
other file needs to change, and evolution starts using it on the next run.

## Requirements

Python 3.7+, `numpy`, `matplotlib`.

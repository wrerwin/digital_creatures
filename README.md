# digital creatures

Little guys on a grid. Nobody tells them what to do — the only rule is who gets
to reproduce, and the behaviour that follows from that is the interesting part.

```bash
uv run python execute.py
```

`uv` reads `pyproject.toml` and `.python-version`, installs Python 3.13 and the
dependencies on first run, and needs no activated virtualenv. Generation 0 is
random noise; watch the survival percentage climb.

## How it works

A generation runs for a fixed number of timesteps. At the end, a **survival
criterion** decides who reproduces — by default, "be in the left fifth of the
world". Survivors are cloned with mutation until the population is full again,
the grid is cleared, and the next generation starts.

Each organism has a **genome**: a fixed-length tuple of genes, where every gene
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
| `settings.py` | `Settings`, a frozen dataclass holding every tunable number |
| `capability_utils.py` | the `Sensor` and `Action` enums — what a creature can perceive and do |
| `brain_utils.py` | genes, mutation, and the network a genome builds |
| `organism.py` | `Organism`, `World`, the generational cycle, and the survival criteria |
| `execute.py` | the runner and the live animation |
| `test_simulation.py` | invariant checks — run `uv run pytest` |
| `sandbox.ipynb` | design notes and a scratchpad for poking at individual creatures |

## Knobs worth turning

```bash
uv run python execute.py --criterion corners   # left, right, centre, corners
uv run python execute.py --watch 0             # skip the animation, just print numbers
uv run python execute.py --watch 5             # animate every 5th generation
uv run python execute.py --seed 42             # repeatable run
```

The survival criterion is the whole selection pressure. Adding one is a
four-line function at the bottom of `organism.py` plus an entry in `CRITERIA`;
the animation works out how to shade the new zone by itself.

Adding a sense or an action is a single enum member in `capability_utils.py`
plus its function — no other file needs to change, and evolution starts using
it on the next run.

`Settings` is frozen, so vary it with `dataclasses.replace` rather than by
assigning to module globals:

```python
from dataclasses import replace
from organism import CRITERIA, World
from settings import Settings

config = replace(Settings(), point_mutation_rate=0.08, n_organisms=500)
world = World(config=config, criterion=CRITERIA["corners"])
```

## Development

```bash
uv sync                 # create the environment
uv run pytest           # 27 invariant checks
uv run ruff check .     # lint
uv run ruff format .    # format
```

`uv.lock` is committed, so `uv sync` reproduces the exact dependency set.

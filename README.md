# digital creatures

Little guys on a grid. Nobody tells them what to do — the only rule is who gets
to reproduce, and the behaviour that follows from that is the interesting part.

```bash
uv run python server.py
```

Then open <http://127.0.0.1:8000>. Or run it from the terminal instead:

```bash
uv run python execute.py
```

`uv` reads `pyproject.toml` and `.python-version`, installs Python 3.13 and the
dependencies on first run, and needs no activated virtualenv. Generation 0 is
random noise; watch the survival percentage climb.

## The web UI

The browser configures a run; the server streams it back a frame at a time.
Alongside the world settings — objective, obstacles, population, generations,
seed — two dropdowns of checkboxes control **what the creatures are allowed to
be**: which senses and which actions evolution may wire up.

Switching a capability off removes it from the search space entirely. No new
gene will target it, so whatever behaviour depended on it has to be found some
other way, or cannot be found at all. Leave only `x_position` and `bias`
against the `left` objective and evolution still solves it, because that is all
the problem needs; take away `pheromone_*` and no amount of running will
produce trail-following.

Changing the controls and pressing Run abandons whatever is in flight and
starts over, so it stays responsive while a long run is going.

Every menu is built from `/api/options`, which reads the enums and registries
directly — a new sense, action, objective or barrier layout appears in the
browser with no front-end change.

## How it works

A generation runs for a fixed number of timesteps. At the end, an **objective**
decides who reproduces. Survivors are cloned with mutation until the population
is full again, the grid is cleared, and the next generation starts.

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

Every organism's wiring can be read back as text, or drawn with `--draw-brain`:

```
border_distance --(-2.13)--> move_x
inner_2 --(+1.44)--> move_forward
```

## What a creature can sense

All readings are held in `[-1, 1]` so no single sense dominates a brain.

| group | senses |
| --- | --- |
| where am I | `x_position`, `y_position`, `border_distance` |
| what is around me | `population_density`, `neighbours_east`, `neighbours_north`, `nearest_neighbour`, `blocked_forward`, `blocked_left`, `blocked_right` |
| what can I smell | `pheromone_here`, `pheromone_east`, `pheromone_north` |
| what was I doing | `last_move_x`, `last_move_y`, `age`, `random`, `bias` |

The `neighbours_*` pair is signed and directional — `+1` means every neighbour
in range lies that way. That is what makes following, flocking and fleeing
reachable at all; `population_density` alone only says "it is crowded here".

## What a creature can do

`move_x`, `move_y`, `move_forward`, `move_left`, `move_random`, `stay`, and
`emit_pheromone`. The action neurons compete rather than take turns: their
levels sum into an urge per axis, resolved probabilistically into one step.

`emit_pheromone` is the only way one creature can change what another
perceives. Scent decays every timestep, so trails fade unless they are
maintained.

## Objectives

The objective is the entire selection pressure, and swapping it is the main
dial for changing what evolves.

| objective | rule |
| --- | --- |
| `left`, `right`, `centre`, `corners` | be inside the region when the generation ends |
| `stay`, `stay-centre` | spend at least half the generation inside the region |
| `there-and-back` | touch the east band, then finish in the west one |
| `top-to-bottom` | touch the north band, then finish in the south one |
| `hazard` | survive a roaming circle that kills whatever it touches |

`stay` is much harsher than `left`: arriving late is worthless. The two-phase
objectives are the first that cannot be solved by a fixed heading — a creature
has to change its mind partway through, which means routing `age` or a
recurrent inner neuron into its movement.

## Obstacles

`--barriers none|wall|slalom|pillars|funnel`. Barriers are solid cells that
block movement and register on the `blocked_*` senses, which turns "head west"
from a complete solution into one that strands a creature in a dead end.

## Things to try

```bash
uv run python execute.py --objective stay --barriers slalom
uv run python execute.py --compare left,stay,corners,hazard   # race them, headless
uv run python execute.py --objective hazard --watch 1         # watch things die
uv run python execute.py --watch 0                            # numbers only, fastest
uv run python execute.py --seed 42                            # repeatable run
```

Keep a creature you like, and start a later run from it:

```bash
uv run python execute.py --objective corners --save-genome good_creature.json
uv run python execute.py --load-genome good_creature.json --draw-brain
```

Genomes are saved as JSON keyed by sense and action *names*, so a file stays
valid after new capabilities are added.

## Extending it

Each of these is a single edit, and nothing else needs to change:

- **a new sense** — a member of `Sensor` plus its function in `capability_utils.py`
- **a new action** — a member of `Action`, plus how it resolves in `Organism.act`
- **a new objective** — a subclass of `Objective` in `objectives.py`; the
  animation draws its zones without being taught about them
- **a new obstacle layout** — a function plus an entry in `barriers.LAYOUTS`

All four show up in the web UI automatically.

`Settings` is frozen, so vary it with `dataclasses.replace` rather than by
assigning to module globals:

```python
from dataclasses import replace
from organism import World
from settings import Settings

config = replace(Settings(), point_mutation_rate=0.08, pheromone_decay=0.99)
world = World(config=config, objective="there-and-back")
```

## Files

| file | what's in it |
| --- | --- |
| `settings.py` | `Settings`, a frozen dataclass holding every tunable number |
| `capability_utils.py` | the `Sensor` and `Action` enums — what a creature can perceive and do |
| `brain_utils.py` | genes, mutation, and the network a genome builds |
| `organism.py` | `Organism`, `World`, the grid layers, and the generational cycle |
| `objectives.py` | what it takes to reproduce |
| `barriers.py` | obstacle layouts |
| `inspect_utils.py` | saving, loading and drawing creatures |
| `execute.py` | the terminal runner, live animation and comparison mode |
| `server.py` | the web UI's backend, streaming runs over a websocket |
| `static/` | the browser front end |
| `test_simulation.py`, `test_server.py` | invariant checks — run `uv run pytest` |
| `sandbox.ipynb` | design notes and a scratchpad for poking at individual creatures |

## Development

```bash
uv sync                 # create the environment
uv run pytest           # 90 invariant checks
uv run ruff check .     # lint
uv run ruff format .    # format
```

`uv.lock` is committed, so `uv sync` reproduces the exact dependency set.

## Known limits

Reproduction is asexual — mutation is the only source of variation, with no
crossover between survivors.

Selection is all-or-nothing: an objective either passes a creature or does not,
with no partial credit. Objectives that almost nobody can satisfy early on
therefore have no gradient to climb, and the population reseeds from random
genomes whenever a generation produces zero survivors. Conjunctive goals need
their two halves chosen so that some creatures get there by luck in the first
few generations.

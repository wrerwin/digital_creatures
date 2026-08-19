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
decides who reproduces. Survivors leave offspring, the grid is cleared, and the
next generation starts — with however many creatures were earned.

## Staying alive is meant to be hard

Three things push back, and together they make the population walk an edge
rather than settle:

**Metabolism.** Creatures burn energy: a base cost each timestep, more when
they move, and — the important one — upkeep for *every distinct sense the brain
is wired to*. A creature reading twelve senses pays for twelve senses whether
or not they earn their keep, so more capability stops being strictly better.
At the default cost a population sheds roughly two of its nine wired senses
over thirty generations. Run out of energy and you starve mid-generation.

**A population that can die.** Nothing is refilled to a fixed size. Each
breeding survivor leaves `offspring_per_survivor` young, capped by the carrying
capacity. At the default of 2.0 the population holds steady at *exactly* 50%
survival — so that line, drawn on the survival chart, is the edge. Below it the
population shrinks, and it can reach zero, which ends the run.

**A shrinking target.** Optionally the survival zone contracts each generation,
so a solution that worked at generation 10 stops working by 50. Worth knowing:
this ratchets the early climb hard, but a population already at carrying
capacity tends to absorb it.

The defaults are deliberately mean. A measured run fell from 250 creatures to
18 before recovering; under sexual reproduction, seeds exist that die out
entirely in three generations.

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

## Reproduction

`--reproduction asexual|sexual`, or the dropdown in the UI.

**Asexual** clones a survivor with mutation. Any survivor breeds, wherever it
ended up, and mutation is the only source of new variation.

**Sexual** makes survivors *find each other*. Two survivors within
`mating_radius` pair up — greedily and monogamously, nearest first — and their
genomes cross over uniformly, every connection coming from one parent or the
other. A survivor with nobody in range leaves nothing at all.

That second clause is the whole point: reaching the zone stops being enough,
because arriving alone is the same as not arriving. It puts real pressure on
the neighbour senses, and it is much harsher — the same seed that cruises
asexually can crash to a handful of creatures, or die out.

Each child takes one parent's lineage at random, so a line can vanish even
while its genes survive in somebody else's descendants.

## Watching the population

Reading one brain tells you what one creature does. The **gene expression**
view tells you what evolution has decided: for every sense and action, the
share of the population that wires it at all.

It is the most informative thing in the UI. A sense at 100% has become
load-bearing; one that falls to 0% has been actively selected away. In one
measured run against the `left` objective, `x_position` went to 100% while
`y_position` and `population_density` were driven to zero — and 200 founding
lineages collapsed to 2.

`--stats` prints the same picture in the terminal.

## Obstacles

`--barriers none|wall|slalom|pillars|funnel`. Barriers are solid cells that
block movement and register on the `blocked_*` senses, which turns "head west"
from a complete solution into one that strands a creature in a dead end.

## Things to try

```bash
uv run python execute.py --objective stay --barriers slalom
uv run python execute.py --reproduction sexual --stats   # mates must be found
uv run python execute.py --zone-shrink 0.02              # ratchet the difficulty
uv run python execute.py --no-metabolism                 # brains cost nothing
uv run python execute.py --compare left,stay,corners     # race them, headless
uv run python execute.py --watch 0                       # numbers only, fastest
uv run python execute.py --seed 42                       # repeatable run
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
| `objectives.py` | what it takes to earn the right to reproduce |
| `reproduction.py` | asexual and sexual strategies, and population dynamics |
| `population_stats.py` | gene expression and lineages across the whole population |
| `barriers.py` | obstacle layouts |
| `inspect_utils.py` | saving, loading and drawing creatures |
| `execute.py` | the terminal runner, live animation and comparison mode |
| `server.py` | the web UI's backend, streaming runs over a websocket |
| `static/` | the browser front end |
| `test_simulation.py`, `test_evolution.py`, `test_server.py` | invariant checks — run `uv run pytest` |
| `sandbox.ipynb` | design notes and a scratchpad for poking at individual creatures |

## Development

```bash
uv sync                 # create the environment
uv run pytest           # 125 invariant checks
uv run ruff check .     # lint
uv run ruff format .    # format
```

`uv.lock` is committed, so `uv sync` reproduces the exact dependency set.

## Known limits

Selection is all-or-nothing: an objective either passes a creature or does not,
with no partial credit. Objectives that almost nobody can satisfy early on
therefore have no gradient to climb, and the population reseeds from random
genomes whenever a generation produces zero survivors. Conjunctive goals need
their two halves chosen so that some creatures get there by luck in the first
few generations.

Mutation is still only point mutation. Genomes cannot grow, shrink, or
duplicate a gene, so brain size is fixed for a whole run.

The shrinking zone ratchets the early climb but does not destabilise a
population that has already reached carrying capacity — measured over 100
generations, a zone contracting from 12% to 4.4% did not dent it.

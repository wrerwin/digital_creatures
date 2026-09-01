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
seed — a **skill dashboard** controls what the creatures are allowed to be.

Skills are split into three collapsible panels, because they answer three
different questions:

| panel | question | holds |
| --- | --- | --- |
| **Sensing** | what can it perceive of the world? | position, walls, neighbours, obstacles, scent |
| **Moving** | what can it do in the world? | the seven actions, including laying scent |
| **Intelligence** | what does it know about itself? | previous movement, age, energy, noise, a constant — plus genome size and inner-neuron count |

Every skill carries a tooltip, and for senses that text is the sensing
function's **own docstring**, so what a user reads cannot drift away from what
the code does. Each panel shows how many of its skills are enabled, and turns
red when a panel is emptied.

Underneath, a running estimate of what the selection costs to run: a brain
wiring all 19 senses burns 0.86 energy per step, about 172 over a 200-step
generation — against a budget of 140, so it starves. Turning senses off is not
only a restriction, it is a discount.

Switching a capability off removes it from the search space entirely. No new
gene will target it, so whatever behaviour depended on it has to be found some
other way, or cannot be found at all. Leave only `x_position` and `bias`
against the `left` objective and evolution still solves it, because that is all
the problem needs; take away `pheromone_*` and no amount of running will
produce trail-following.

Changing the controls and pressing Run abandons whatever is in flight and
starts over, so it stays responsive while a long run is going.

**Recall.** Every frame of the generation currently running is kept, so when a
run finishes — or you stop it — the last complete generation can be replayed
and scrubbed frame by frame under the world view. It parks on the final frame,
which is the state selection actually acted on, so you can wind back and watch
how the survivors got there.

Every menu is built from `/api/options`, which reads the enums and registries
directly — a new sense, action, objective or barrier layout appears in the
browser, in the right panel and correctly explained, with no front-end change.
A sense's panel comes from `SENSOR_CATEGORY` in `capability_utils.py`.

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
breeding survivor leaves offspring, capped by the carrying capacity, and the
population can reach zero — which ends the run.

The rate is **density-dependent**, decaying geometrically from 14 offspring per
survivor in an empty world to 1.15 at the carrying capacity. So the bar for
holding your ground climbs as the world fills — from 7% survival when the
population has crashed to 87% when it is full. Cheap to recover, expensive to
stay full.

A population therefore settles wherever its survival rate meets that rising
bar, and drifts as survival does. The dashed line on the survival chart is the
bar itself: where the survival line crosses it is exactly where the population
turns around.

Measured across every objective — three seeds each, 30 generations, capacity
400, sampled from generation 10 once the population has settled:

| objective | population range | swing | generations at the cap |
| --- | --- | --- | --- |
| `there-and-back` | 213 – 400 | 154 | 7% |
| `top-to-bottom` | 217 – 400 | 135 | 2% |
| `hazard` | 267 – 400 | 123 | 38% |
| `corners` | 285 – 400 | 111 | 40% |
| `stay-centre` | 179 – 311 | 106 | 0% |
| `centre` | 182 – 329 | 102 | 0% |
| `stay` | 392 – 400 | 4 | 92% |
| `left` | 393 – 400 | 4 | 95% |
| `right` | 382 – 400 | 6 | 92% |

Nothing went extinct in any of the 27 runs, and every objective dips hard in
the first generation or two — as low as 62 — before climbing back out.

**Three objectives still pin: `left`, `right` and `stay`.** Their evolved
survival exceeds even the 87% demanded at capacity, so nothing stops them
filling the world. That is the honest limit of this mechanism: an objective a
population solves *completely* will always saturate, and the fluctuation lives
in the six that stay hard. To get movement on an easy one, raise "offspring
when full" toward 1.0 until even that survival rate is marginal.

**A shrinking target.** Optionally the survival zone contracts each generation,
so a solution that worked at generation 10 stops working by 50. Worth knowing:
this ratchets the early climb hard, but a population already at carrying
capacity tends to absorb it.

The defaults are deliberately mean. A measured run fell from 250 creatures to
18 before recovering; under sexual reproduction, seeds exist that die out
entirely in three generations.

### Difficulty is a ladder, not a cliff

One `survival_zone_fraction` cannot mean the same thing to every objective. A
circle of radius `0.12w` covers 4.4% of the grid where a band of width `0.12w`
covers 12%, so the same setting made some objectives comfortable and others
impossible. Each objective now scales that setting by its own `zone_scale`.

Area is not the whole story either — reaching a point in the middle is a harder
thing for a brain to *compute* than heading in one fixed direction, so `centre`
is given more room than equal area would suggest. What is tuned against is
measured survival of an **unevolved** population, which is what decides whether
a run can get started at all:

| objective | unevolved survival |
| --- | --- |
| `hazard` | 40% |
| `stay-centre` | 28% |
| `corners` | 24% |
| `stay` | 24% |
| `left` | 22% |
| `right` | 17% |
| `centre` | 15% |
| `there-and-back` | 11% |
| `top-to-bottom` | 8% |

All of them clear the 6.25% a struggling population needs, so every objective
is winnable — while still spanning a five-fold range in how hard it is.

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

**[BRAIN.md](BRAIN.md) explains the whole mechanism** — how the network is put
together, how a genome becomes behaviour in one timestep, how the senses feed
into it, and what evolution is actually doing across generations.

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
| `stay`, `stay-centre` | spend at least a third of the generation inside the region |
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
load-bearing; one that falls to 0% has been actively selected away.

**Sweep the pointer across the bars** to see how any one capability got there —
its whole share-per-generation history, with where it started, where it peaked
and where it ended. Click a bar to pin it so the pointer can go elsewhere.
This is where the story is: in one measured run, `x_position` fell from 45% to
3% while `population_density` climbed from 33% to 67%, which is the population
switching from navigating by absolute position to navigating by its neighbours.

**Lineages** are counted against the founders, not in isolation — five lineages
left is unremarkable out of ten and a near-total collapse out of two hundred.
The readout and the purple line on the chart both show the ratio. Runs
routinely end with 2–3% of founding lines still present.

`--stats` prints the same picture in the terminal:

```
population 156   lineages 5 of 150 (3% remaining)   mean senses wired 8.1
```

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
| `BRAIN.md` | how the neural network, the senses and the evolution actually work |
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
uv run pytest           # 156 invariant checks
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

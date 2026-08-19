# World geometry. Positions are integer grid cells in [min, max).
x_min = 0
x_max = 100
y_min = 0
y_max = 100

# Population and timing
n_organisms = 250
steps_per_generation = 200
n_generations = 100

# Brain structure
n_inner_neurons = 4      # internal neurons available as connection sources/sinks
n_genes = 24             # connections per genome
max_weight = 4.0         # gene weights are drawn from [-max_weight, max_weight]

# Reproduction
point_mutation_rate = 0.02   # per-gene chance of a mutation during reproduction
weight_jitter = 0.4          # stddev of the nudge applied to a mutated weight

# Selection: fraction of the world's width that counts as the survival zone.
# Organisms whose x is inside this band at the end of a generation reproduce.
survival_zone_fraction = 0.2

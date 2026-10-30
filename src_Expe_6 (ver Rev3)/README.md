# Experiment 6

Unbiased hypervolume estimator on a simple **continuous multi-objective linear
problem** (weighted simplex), for which the exact hypervolume is known in closed
form. Implemented in Julia with multi-threaded parallelisation.

## Problem

```
max   z_k = a_k * x_k        k = 1..d
s.t.  sum_i b_i * x_i <= c
      x_i >= 0  (continuous)
```

Reference point: the origin. The dominated region is the simplex with vertices
`z_k = a_k * c / b_k` on the axes, hence

```
H = prod_k (a_k * c / b_k) / d!
```

For a direction ψ (λ = 1/ψ), the closed form of the subproblem is
`L(ψ) = c / sum_k (b_k * ψ_k / a_k)`.

Instances: `a = b = c = 1` (`"unit"`, `H = 1/d!`) or integers drawn in
`[1, 100]` (`"rand"`); the script runs both.

## Project structure

```
src_Expe_6/
├── mainExp6.jl                  # entry point
├── README.md
├── .gitignore
└── src/
    ├── analyze.jl               # statistical utilities
    ├── common.jl                # ψ! and λ! (directions)
    ├── instance_LP.jl           # instance structure, generator, exact H, closed-form L
    └── estimHyperVol_LP.jl      # estimator: L_threaded (LP), L_analytic_threaded, Hrevised3
```

## Prerequisites

Julia ≥ 1.9 with the packages
`JuMP`, `Gurobi`, `Distributions`, `SpecialFunctions`, `HypothesisTests`,
`Statistics`, `Plots`. No C library is needed. Any LP solver supported by
JuMP can replace Gurobi (change `solver` in `mainExp6.jl`).

## Run

From a shell, in the folder `src_Expe_6/`:

```bash
julia --threads 8 mainExp6.jl
```

## Configuration (top of `mainExp6.jl`)

| Parameter | Meaning |
|---|---|
| `listd` | numbers of objectives d swept (= number of variables) |
| `listmodes` | `"unit"`: a = b = c = 1; `"rand"`: integers in [1, `maxval`] |
| `nInstances` | number of instances / repetitions |
| `listnbrWeights` | numbers of directions N tested |
| `solver` | LP solver |
| `solver_fn` | `L_threaded` (LP) or `L_analytic_threaded` (closed form) |

## Output files

| File | Description |
|---|---|
| `expe6-<mode>-<d>obj.res` | detailed results per instance, for each mode and d |
| `expe6-<mode>-tableResults.res` | LaTeX-ready summary table over (d, N), overwritten at each run |
| `expe6-<mode>-err.png` | average relative error vs N, one curve per d (log-log) |
| `expe6-<mode>-CI.png` | average 95% CI width vs N, one curve per d (log-log) |

`<mode>` is `unit` (a = b = c = 1) or `rand` (random integers in [1, 100]).

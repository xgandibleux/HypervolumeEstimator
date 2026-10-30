# Experiment 4 — parallel implementation (moUKP)

Unbiased estimator of the hypervolume of the true Pareto front, applied to the
**0-1 multi-objective knapsack problem (moUKP)**, with the parallel implementation: the N directions
are split into chunks of fixed size `CHUNKSIZE = 50`, each chunk having its own model and its own
random generator seeded by the chunk index (`1234 + chunk_id`). The sampled directions, hence the
estimates, do not depend on the number of threads.

The reference point is the improved lower bound on the nadir point `r_*^{nMin}`
(`reference_point_LB` in `src/instance_UKP.jl`): `nMin` is obtained by sorting the items by
decreasing weight and selecting them until the **first** item that does not fit.

The experiment has two parts, each with its own entry point:

| Part | Entry point | Question |
|---|---|---|
| 1 | `mainExp4_50chunks_comparison-revT.jl` | speedup of 8 threads vs 1 thread; are the estimates identical? |
| 2 | `mainExp4_2x2_modes-revT.jl` | orchestration scheme (1 chunk / 50 chunks) x Gurobi threading policy (1 thread / free) |

## Content

```
src_Expe_4/
├── mainExp4_50chunks_comparison-revT.jl   # entry point, part 1
├── mainExp4_2x2_modes-revT.jl             # entry point, part 2
├── build_lib.sh / build_lib.bat           # compile the fpli_hv library (macOS-Linux / Windows)
├── src/
│   ├── analyze.jl                         # average_value
│   ├── common-revT.jl                     # directions ψ, weights λ, CHUNKSIZE, solver settings
│   ├── estimHyperVol_UKP-revT.jl          # L_threaded, Hrevised3 (estimator, CI 95%)      [parts 1 and 2]
│   ├── modes-revT.jl                      # L_modes (modes A-D)                            [part 2]
│   ├── files.jl                           # exact hypervolume H(Y_N) via fpli_hv (@ccall)
│   ├── instance_UKP.jl                    # instance, random generator, reference_point_LB
│   ├── solve_UKP.jl                       # exact set Y_N (Tamby-Vanderpooten via MOA)
│   └── libfpli_hv.dylib                   # fpli_hv compiled for macOS
└── src HV/hv-1.3-src/                     # C source of fpli_hv (Fonseca, Paquete, Lopez-Ibanez)
```

## Prerequisites

Julia >= 1.9 with `JuMP`, `Gurobi`, `MultiObjectiveAlgorithms`, `Distributions`,
`SpecialFunctions`, `HypothesisTests`, `Statistics`, `Plots`, `DelimitedFiles`; a Gurobi licence;
a C compiler for the library (not needed on macOS if `src/libfpli_hv.dylib` is used).

Library, once (Linux; macOS if the `.dylib` does not load):

```bash
bash build_lib.sh
```

All commands below are shell commands, run from the folder `src_Expe_4/` (the folder containing the
entry points). The output files are written in the current folder. The paper configurations
(`n` variables, `o` objectives) are `25 4`, `50 3` and `100 2`.

## Part 1 — 1 thread vs 8 threads

```bash
julia --threads 1 mainExp4_50chunks_comparison-revT.jl 25 4
julia --threads 8 mainExp4_50chunks_comparison-revT.jl 25 4
```

Repeat for `50 3` and `100 2`. Without arguments the script uses `25 4`. The second run compares its
estimates with those of the first one (file `ukp-verif-n-o-T*.txt`) and prints `VERIFICATION ... OK`
if they are bitwise identical.

| Output file | Content |
|---|---|
| `ukp-threads-n-o-T*.res` | detailed results per instance |
| `ukp-tableThreadsComparison-T*.res` | LaTeX table (N, norm, CI width, relative error, time) |
| `ukp-verif-n-o-T*.txt` | normalised estimates, used for the cross-thread check |
| `ukp-threads-err-n-o-T*.png`, `ukp-threads-time-n-o-T*.png` | average relative error, average time |

## Part 2 — orchestration scheme x Gurobi threading policy (modes A, B, C, D)

| Mode | Chunks | Julia threads | Gurobi |
|---|---|---|---|
| A | 1 | 1 | constrained to 1 thread |
| B | 1 | 1 | free |
| C | 50 | 8 | constrained to 1 thread |
| D | 50 | 8 | free |

The four modes use the same instances as part 1 and the same sampled directions: only the elapsed
time differs. Paper configuration: `50 3`.

```bash
julia --threads 1 mainExp4_2x2_modes-revT.jl 50 3     # modes A and B
julia --threads 8 mainExp4_2x2_modes-revT.jl 50 3     # modes C and D
```

The script prints `CHECK` lines: maximal difference between the estimates of the modes (about 1e-10,
solver tolerances); mode C is also compared with part 1 (file `ukp-verif-50-3-T8.txt`, if present in
the same folder; expected difference 0). Once the four modes are done, `ukp-modes-table-50-3.res`
contains the elapsed times and the speedups t_A/t_B, t_A/t_C and t_D/t_C.

| Output file | Content |
|---|---|
| `ukp-modes-norm-n-o-<mode>.txt` | normalised estimates of the mode |
| `ukp-modes-time-n-o-<mode>.txt` | average elapsed time of the mode, per N |
| `ukp-modes-table-n-o.res` | LaTeX table of the four modes and the speedups |

Do not run other heavy jobs at the same time: the elapsed times are the result.

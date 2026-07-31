# Experiments 3 / 4 / 5

Unbiased hypervolume estimator for the **0-1 Multi-Objective Knapsack Problem (UKP)**
and the **Uncapacitated Facility Location Problem (UFLP)**, implemented in Julia
with multi-threaded parallelisation.

## Project structure

```
HVfinal/
├── mainExp1_UKP.jl       # entry point — UKP experiment
├── mainExp1_UFLP.jl      # entry point — UFLP experiment
├── build_lib.sh          # compile fpli_hv shared library (macOS / Linux)
├── build_lib.bat         # compile fpli_hv shared library (Windows / MinGW-w64)
└── src/
    ├── common.jl         # shared functions (ψ!, λ!)
    ├── instance_UKP.jl   # UKP instance structure and generators
    ├── instanceUFLP.jl   # UFLP instance structure and generators
    ├── solve_UKP.jl      # exact UKP solver (MOA / Tamby-Vanderpooten)
    ├── solveUFLP.jl      # exact UFLP solver (MOA / Tamby-Vanderpooten)
    ├── estimHyperVol_UKP.jl   # hypervolume estimator — UKP
    ├── estimHyperVol_UFLP.jl  # hypervolume estimator — UFLP
    ├── files.jl          # I/O and hypervolume measurement (@ccall to fpli_hv)
    ├── analyze.jl        # statistical utilities
    └── src HV/hv-1.3-src/    # C source of the fpli_hv library
```

## Prerequisites

- **Julia** ≥ 1.9 with the following packages:
  `JuMP`, `HiGHS`, `MultiObjectiveAlgorithms`, `Distributions`,
  `SpecialFunctions`, `HypothesisTests`, `Statistics`, `Plots`
- A **C compiler** to build the `fpli_hv` shared library (see below)

## Step 1 — Build the shared library

The hypervolume measurement uses `fpli_hv` (Fonseca, Paquete & López-Ibáñez, 2006)
called directly via Julia's `@ccall`. It must be compiled once before the first run.

### macOS

```bash
bash build_lib.sh
```

Requires Apple Command Line Tools (`xcode-select --install`).
Produces `src/libfpli_hv.dylib`.

### Linux

```bash
bash build_lib.sh
```

Requires `gcc` or `clang` (`sudo apt install gcc` on Debian/Ubuntu).
Produces `src/libfpli_hv.so`.

### Windows

Requires [MinGW-w64](https://www.mingw-w64.org/) with `gcc` available in `PATH`.

```bat
build_lib.bat
```

Produces `src\fpli_hv.dll`.
Alternatively, use WSL and follow the Linux instructions.

## Step 2 — Run the experiments

Launch Julia with 8 threads (optimal on Apple M2 Pro; adjust to your machine):

```bash
# UKP experiment
julia --threads 8 mainExp1_UKP.jl

# UFLP experiment
julia --threads 8 mainExp1_UFLP.jl
```

## Configuration

Edit the parameters at the top of each entry point:

| Parameter | UKP | UFLP |
|---|---|---|
| Number of items / users | `n` | `nI`, `nJ` |
| Number of objectives | `o` | `o` |
| Number of instances | `nInstances` | `nInstances` |
| MIP solver | `solver` | `solver` |
| Estimator strategy | `solver_fn` | — |

## Output files

All output files are prefixed with the problem name:

| File | Description |
|---|---|
| `ukp-n-o.res` / `uflp-nIxnJ-o.res` | detailed results per instance |
| `ukp-tableResultsExpe1.res` / `uflp-tableResultsExpe1.res` | LaTeX-ready summary table |
| `ukp-n-o.png` / `uflp-nIxnJ-o.png` | average relative error curve |
| `ukp-H-n-o.png` / `uflp-H-nIxnJ-o.png` | normalised H estimated with 95% CI |

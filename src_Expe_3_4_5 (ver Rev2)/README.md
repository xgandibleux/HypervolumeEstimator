# Experiments 3 / 4 / 5

Unbiased hypervolume estimator for the **0-1 Multi-Objective Knapsack Problem (UKP)**
and the **Uncapacitated Facility Location Problem (UFLP)**, implemented in Julia
with multi-threaded parallelisation.

## Project structure

```
src_Expe_3_4_5 (ver Rev2)/
├── mainExp1_UKP.jl                         # (Refactoring of Expe 1 in progress)
├── mainExp1_UKP_10obj_7weights.jl          # (a variant of the refactoring of Expe 1 in progress)
├── mainExp1_UKP_large.jl                   # (a variant of the refactoring of Expe 1 in progress)
├── mainExp3_rp_comparison.jl               # entry point — experiment 3 (reference point comparison)
├── mainExp4_8threads_comparison.jl         # (a variant of Expe 4 with 8 threads=chunks)
├── mainExp4_50chunks_comparison-revT.jl    # entry point — experiment 4 (50 chunks, thread comparison)
├── mainExp5_8threads_UFLP.jl               # (a variant of Expe 5 with 8 threads=chunks)
├── mainExp5_50chunks_UFLP-revT.jl          # entry point — experiment 5, UFLP with 50 chunks
├── build_lib.sh                            # compile fpli_hv shared library (macOS / Linux)
├── build_lib.bat                           # compile fpli_hv shared library (Windows / MinGW-w64)
├── .gitignore
├── README.md
├── src/
│   ├── analyze.jl                          # statistical utilities
│   ├── common.jl                           # shared functions
│   ├── common-revT.jl                      # shared functions (revT variant)
│   ├── common-revT v0.jl.                  # (old version)
│   ├── computeCI.jl                        # confidence interval computation
│   ├── estimHyperVol_UFLP.jl               # hypervolume estimator — moUFLP
│   ├── estimHyperVol_UFLP-revT.jl          # hypervolume estimator — moUFLP (revT variant)
│   ├── estimHyperVol_UKP.jl                # hypervolume estimator — moUKP
│   ├── estimHyperVol_UKP-revT.jl           # hypervolume estimator — moUKP (revT variant)
│   ├── estimHyperVol_UKP-revT v0.jl        # (old version)
│   ├── estimHyperVol_UKP_nonorm.jl         # hypervolume estimator — moUKP (non-normalised variant)
│   ├── files.jl                            # I/O and hypervolume measurement (@ccall to fpli_hv)
│   ├── hv                                  # compiled hypervolume executable
│   ├── instanceUFLP.jl                     # moUFLP instance structure and generators
│   ├── instance_UKP.jl                     # moUKP instance structure and generators
│   ├── libfpli_hv.dylib                    # compiled fpli_hv shared library (macOS)
│   ├── solveUFLP.jl                        # exact moUFLP solver
│   ├── solve_UKP.jl                        # exact moUKP solver
│   └── .gitignore
└── src HV/hv-1.3-src/                      # C source of the fpli_hv library
```

## Prerequisites

- **Julia** ≥ 1.9 with the following packages:
  `JuMP`, `Gurobi`, `MultiObjectiveAlgorithms`, `Distributions`,
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

## Output files

All output files are prefixed with the problem name:

| File | Description |
|---|---|
| `ukp-n-o.res` / `uflp-nIxnJ-o.res` | detailed results per instance |
| `ukp-tableResultsExpe1.res` / `uflp-tableResultsExpe1.res` | LaTeX-ready summary table |
| `ukp-n-o.png` / `uflp-nIxnJ-o.png` | average relative error curve |
| `ukp-H-n-o.png` / `uflp-H-nIxnJ-o.png` | normalised H estimated with 95% CI |

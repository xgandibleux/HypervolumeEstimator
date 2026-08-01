# Experiments 1 / 2

Unbiased hypervolume estimator for the **0-1 Multi-Objective Knapsack Problem (UKP)**, implemented in Julia.

## Project structure

```
src_Expe_1_2 (ver Rev1)/
├── mainExp1.jl               # entry point — experiment 1
├── mainExp1Bis.jl            # code for generating a graphic
├── mainExp1Ter.jl            # code for generating a graphic
├── mainExp21.jl              # entry point — experiment 2.1
├── mainExp22.jl              # entry point — experiment 2.2
├── README.md
└── src/
    ├── analyze.jl            # statistical utilities
    ├── approxMO01UKP.jl      # using MetaJul for computing an approximation of $Y_N$ with NSGA-II
    ├── computeCI.jl          # confidence interval computation
    ├── estimHyperVol1.jl     # hypervolume estimator — moUKP
    ├── files.jl              # I/O on files
    ├── hv                    # compiled hypervolume executable
    ├── instanceMO01UKP.jl    # moUKP instance structure and generators
    ├── mainH.jl              # estimation outside experiment
    └── solveMO01UKP.jl       # exact moUKP solver
```

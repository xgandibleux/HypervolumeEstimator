# Hypervolume Estimator
Algorithm for computing a consistent and unbiased estimation of the hypervolume of the set of nondominated points a priori unknown.
For the numerical experimentation needs, the multi-objective optimization problem implemented in the current version of the code is the 01 unidimensional knapsack problem.

## Paper
Available on *to add on [optimization-online](https://optimization-online.org/2025/09/consistent-and-unbiased-estimation-of-the-hypervolume-of-an-unknown-true-pareto-front/)* 

Citate this work: *Consistent and unbiased estimation of the hypervolume of an unknown true Pareto front. Xavier Gandibleux and Andrzej Jaszkiewicz. Optimization-online.
Published: 2025/09/03*

## Acknowledgement
A code for measuring the hypervolume value of a set of nondominated points is required.
We use the code `hv` (version 1.3) available online [here](https://lopez-ibanez.eu/hypervolume) and cloned on this repository (see in folder `src HV`).
An executable version named `hv` has to be present into the folder `src` to perform properly a numerical experiment with our estimation algorithm.
Follow the indications provided [here](https://lopez-ibanez.eu/hypervolume) to compile the `hv` code.

## Install and run the code the first time
- download all the repository from GitHub
- in a terminal, move inside the directory downloaded 
- compile `hv` on your computer and move the exec file into the `src` folder 
(on mac: make OPT_CFLAGS="-O2 -g")  
- invoke `julia`
- in the REPL, invoke e.g. `include("mainExp1.jl")`

Tested on macBook Pro under macOS v14.6, with Julia 1.10 using packages 
- JuMP.jl v1.26.0
- GLPK.jl v1.2.1
- MultiObjectiveAlgorithms.jl v1.4.3
- Distributions.jl v0.25.120
- SpecialFunctions.jl v2.5.1
- HypothesisTests.jl v0.11.5
- Plots.jl v1.40.14
- MetaJul.jl v0.2.0 (https://github.com/jMetal/MetaJul)

## Setup the number of variables and the number of objectives
Change in the code the value assigned to `n` and `o`.
Currently `n=10` and `o=3`.

## Numerical experiments available
- Experiment 1. Given one instance, compute $Y_N$ and $H(Y_N)$ vs estimation $\tilde{H}$ for 20 trials/7 sets of weights (from 100 to 10000 weights);
   returns $H(Y_N)$, the average absolute and relative error on $\tilde{H}$, interval confidence of value 95%, elapsed times.
- Experiment 2.1 Given one instance, compute $Y_{A_1}$ and $H(Y_{A_1})$ vs estimation $\tilde{H}$ for 20 trials/7 sets of weights (from 100 to 10000 weights);
   returns $H(Y_{A_1})$, the average absolute and relative error on $\tilde{H}$, interval confidence of value 95%, elapsed times.
- Experiment 2.2 Given one instance, compute $Y_{A_2}$ and $H(Y_{A_2})$ vs estimation $\tilde{H}$ for 20 trials/7 sets of weights (from 100 to 10000 weights);
   returns $H(Y_{A_2})$, the average absolute and relative error on $\tilde{H}$, interval confidence of value 95%, elapsed times.
- Experiment 3. Given 20 instances with $n$ and $d$ fixed, compute $Y_N$ and $H(Y_N)$ vs estimation $\tilde{H}$ for 1 trial/1 sets of weights (2000 weights);
   returns average absolute and relative error on $\tilde{H}$, average elapsed times.
- Experiment 4. Given 1 instance with $n$ and $d$ fixed, compute the estimation $\tilde{H}$ for 1 trial/1 sets of weights (2000 weights);
   returns average elapsed times.

## Data and outputs
Instances are generated on the fly (the seed is fixed to `1234`). A run displays in the terminal the results and saves on files (for exp1, exp2.1 and exp2.2)
- the instance generated
- the set of nondominated points
- a full trace of the resolution
- two figures (the average relative error; values computed for H, H estimated, CI)

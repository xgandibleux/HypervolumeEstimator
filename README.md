# Hypervolume Estimator
Algorithm for computing a consistent and unbiased estimation of the hypervolume of the set of nondominated points a priori unknown.
For the numerical experimentation needs, the multi-objective optimization problem implemented in the current version of the code is the 01 unidimensional knapsack problem (experiments 1 to 4) and the 01 uncapacited facility location problem (experiment 5). Gurobi is used as MIP solver, and HV for computing the hypervolume indicator.


## Paper
Available on [optimization-online](https://optimization-online.org/2025/09/consistent-and-unbiased-estimation-of-the-hypervolume-of-an-unknown-true-pareto-front/)

Citate this work: *Consistent and unbiased estimation of the hypervolume of an unknown true Pareto front. Xavier Gandibleux and Andrzej Jaszkiewicz. Optimization-online.
Published: 2025/09/03*

## Note
The codes have evolved as work on the paper progressed. A refactoring is currently underway. As it currently stands, the `Expe_1_2` folder contains the code corresponding to the results of Experiments 1 and 2 in the paper. The same applies to the `Expe_3_4_5` folder and Experiments 3, 4, and 5.

## Numerical experiments available
- **Experiment 1. moUKP:** Given 20 instances, compute $Y_N$ and $H(Y_N)$ vs estimation $\tilde{H}$ for 1 trial/7 sets of weights (from 100 to 10000 weights);
   returns $H(Y_N)$, the average absolute and relative error on $\tilde{H}$, interval confidence of value 95%, elapsed times.
- **Experiment 1.bis moUKP:** Given 20 instances with $n$ and $d$ fixed, compute $Y_N$ and $H(Y_N)$ vs estimation $\tilde{H}$ for 1 trial/1 sets of weights (2000 weights);
   returns average absolute and relative error on $\tilde{H}$, average elapsed times.
- **Experiment 1.ter moUKP:** Given 1 instance with $n$ and $d$ fixed, compute the estimation $\tilde{H}$ for 1 trial/1 sets of weights (2000 weights);
   returns average elapsed times.

- **Experiment 2.1 moUKP:** Given one instance, compute $Y_{A_1}$ and $H(Y_{A_1})$ vs estimation $\tilde{H}$ for 20 trials/7 sets of weights (from 100 to 10000 weights);
   returns $H(Y_{A_1})$, the average absolute and relative error on $\tilde{H}$, interval confidence of value 95%, elapsed times.
- **Experiment 2.2 moUKP:** Given one instance, compute $Y_{A_2}$ and $H(Y_{A_2})$ vs estimation $\tilde{H}$ for 20 trials/7 sets of weights (from 100 to 10000 weights);
   returns $H(Y_{A_2})$, the average absolute and relative error on $\tilde{H}$, interval confidence of value 95%, elapsed times.

 - **Experiment 3. moUKP:** Given 3 problem configurations (10 instances each), compute $\tilde{H}$ for both reference points $r_•^{0}$ and $r_•^{nMin}$, same seed, 7 sets of weights (from 100 to 10000 weights); returns the normalised estimate $\tilde{H}/H_{\text{exact}}$, the relative error, and the 95\% confidence interval width, for each reference point.

- **Experiment 4. moUKP:** Given 3 problem configurations (10 instances each), compute $\tilde{H}$ with 1 thread and with 8 threads, 50 chunks, same seed, 7 sets of weights (from 100 to 10000 weights); returns the elapsed time for each thread configuration, the normalised estimate $\tilde{H}/H_{\text{exact}}$, the 95\% confidence interval width, and the relative error.

- **Experiment 5. moUFLP:** Given 8 problem configurations (10 instances each), compute the exact nondominated set $Y_N$ and the estimate $\tilde{H}$ with 8 threads, 7 sets of weights (from 100 to 10000 weights); returns the elapsed time for the exact computation and for the estimator, the normalised estimate $\tilde{H}/H_{\text{exact}}$, the 95\% confidence interval width, and the relative error.  


## Environment
Tested on macBook Pro under macOS v14.6, with Julia 1.12 using packages 
- JuMP.jl v1.30.1
- Gurobi v1.9.2
- MultiObjectiveAlgorithms.jl v1.12.0
- Distributions.jl v0.25.129
- SpecialFunctions.jl v2.8.0
- HypothesisTests.jl v0.11.8
- Plots.jl v1.41.6
- MetaJul.jl v0.3.0 (https://github.com/jMetal/MetaJul)


## Acknowledgement
1. A code for measuring the hypervolume value of a set of nondominated points is required.
We use the code `hv` (version 1.3) available online [here](https://lopez-ibanez.eu/hypervolume) and cloned on this repository (see in folder `src HV`).
An executable version named `hv` has to be present into the folder `src` to perform properly a numerical experiment with our estimation algorithm.
Follow the indications provided [here](https://lopez-ibanez.eu/hypervolume) to compile the `hv` code.

2. The Gurobi MIP solver is used by the codes to solve subproblems. A properly installed version with the required license is necessary. JuMP can also work with other MIP solvers, including HiGHS. It is the user's responsibility to adapt the instructions in JuMP to switch from Gurobi to HiGHS if the latter is the chosen solver.

  
## Install the code the first time
- download all the repository from GitHub
- in a terminal, move inside the directory downloaded 
- compile `hv` on your computer and move the exec file into the `src` folder 
(on mac: make OPT_CFLAGS="-O2 -g")  

## Setup the number of variables and the number of objectives
Change in the code the value assigned to `n` and `o`.
Currently `n=10` and `o=3`.

## Run the code
**for experiments 1 to 3:**

option 1:
- invoke `julia`
- in the REPL, enter the following command: `include("mainExp1.jl")`
   
option 2:
- in a terminal, enter the following command: e.g. `julia --threads 8 mainExp1.jl`

**for experiments 4 and 5:**
- in a terminal, enter the following command: e.g. `julia --threads 8 mainExp1.jl`

## Data and outputs
Instances are generated on the fly (the seed is fixed to `1234`). A run displays in the terminal the results and saves on files (for exp1, exp2.1 and exp2.2)
- the instance generated
- the set of nondominated points
- a full trace of the resolution
- two figures (the average relative error; values computed for H, H estimated, CI)

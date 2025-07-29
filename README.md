# Hypervolume Estimator
Algorithm for computing a consistent and unbiased estimation of the hypervolume of an unknown image of Pareto front.
For the numerical experimentation needs, the multi-objective optimization problem implemented in the current version of the code is the 01 unidimensionnal knapsack problem.

## Paper
Available on *to add on [optimization-online](https://optimization-online.org/)* 

Citate this work: *reference to add*

## Acknowledgement
A code for measuring the hypervolume value of a set of nondominated points is required.
We use the code `hv` (version 1.3) available online [here](https://lopez-ibanez.eu/hypervolume) and cloned on this repository (see in folder `src HV`).
An executable version named `hv` has to be present into the repository `src` to perform properly a numerical experiment with our estimation algorithm.
Follow the indications provided [here](https://lopez-ibanez.eu/hypervolume) to compile the `hv` code.

## Install and run the code the first time
- download all the repository from GitHub
- in a terminal, move inside the directory downloaded 
- compile `hv` on your computer and move the exec file into the `src` folder
- invoke `julia`
- in the REPL, invoke `include("main.jl")`

Tested on macBook Pro under macOS v14.6, with Julia 1.10 using packages 
- JuMP.jl v1.26.0
- GLPK.jl v1.2.1
- MultiObjectiveAlgorithms.jl v1.4.3
- Distributions.jl v0.25.120
- SpecialFunctions.jl v2.5.1
- HypothesisTests.jl v0.11.5
- Plots.jl v1.40.14

## Data and outputs
Instances are generated on the fly. A run displays in the terminal the results and saves on files
- the instance generated
- the set of nondominated points
- a full trace of the resolution
- a figure of the average relative error

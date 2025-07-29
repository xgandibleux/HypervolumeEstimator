# Hypervolume Estimator

## Paper
Available on *to add on [optimization-online](https://optimization-online.org/)* 

How to citate:
- *to add*

## Data
Instances are generated on the fly.

## Acknowledgement
A code for measuring the hypervolume value of a set of nondominated points is required.
We use the code `hv` (version 1.3) available online [here](https://lopez-ibanez.eu/hypervolume) and cloned on this repository (see in folder `src HV`).
An executable version named `hv` has to be present into the repository `src` to perform properly a numerical experiment with our estimation algorithm.
Please follow the indications provided [here](https://lopez-ibanez.eu/hypervolume) to compile the `hv` code.

## Install and run the code the first time
- download all the repository from GitHub
- in a terminal, move inside the directory downloaded 
- compile `hv` on your computer and move the exec file into the `src` folder
- invoke `julia`
- in the REPL, invoke `include("main.jl")`

#= EXPERIMENT 4: 
  - with 100≤n≤5000 and 2≤d≤10 do 
    - for `n` and `d` given 
    - generate `trial` instances randomly  
  - for each instance 
    - estimate H for a set of 2000 weights
    - get elapsed times 
  - report average value of
    - elapsed time for H estimated for n and d given  
=#

using Printf
using Random
using JuMP, Gurobi
import MultiObjectiveAlgorithms as MOA
using SpecialFunctions
using HypothesisTests
using Statistics
using Base.Threads

Random.seed!(1234)

include("src/common.jl")
include("src/instance_UKP.jl")
include("src/solve_UKP.jl")
include("src/files.jl")
include("src/estimHyperVol_UKP.jl")
include("src/analyze.jl")

# =============================================================================
solver    = Gurobi.Optimizer
solver_fn = L_threaded

numberOfWeights = 2000   # number of directions
trials          = 3      # number of instances generated

allCPUt  = Matrix{Float64}(undef, 11, 11)
global nLines = 1

nVar = [100, 1000]
nObj = [2, 10]

for n in nVar

    global nCol = 1
    for o in nObj

        # =============================================================================
        println("-"^80)
        println("  number of variables  : ", n)
        println("  number of objectives : ", o)
        println("  solver MIP invoked   : ", solver)
        println("  threads              : ", Threads.nthreads(), "\n")

        listCPUt = Float64[]

        for _ in 1:trials

            # generate instance and compute reference point
            inst = generate_MO01UKP(n, o)
            rp   = reference_point_LB(inst)

            # estimate H
            print("    Compute H̃ : ")

            # Hexact not available here — pass 1.0 as placeholder
            # (only H_estimated is used, not the normalized value)
            startH    = time()
            H̃, _      = Hrevised3(1.0, solver, inst, rp, numberOfWeights, solver_fn)
            t_elapsedH = round(time() - startH, digits=2)

            @printf("    H estimated with N=%d: %1.6e  (%.2f s)\n",
                numberOfWeights, H̃, t_elapsedH)
            push!(listCPUt, t_elapsedH)

        end

        avCPUt = average_value(listCPUt)
        @printf("  average CPUt for H̃ = %.2f s\n", avCPUt)

        allCPUt[nLines, nCol] = avCPUt
        global nCol += 1
    end
    global nLines += 1
end


using Plots

cput = deepcopy(allCPUt)

plot(nVar,  cput[1:length(nVar), 1], label="d = "*string(nObj[1]), lw=2, marker=:circle)
plot!(nVar, cput[1:length(nVar), 2], label="d = "*string(nObj[2]), lw=2, marker=:square)
plot!(nVar, cput[1:length(nVar), 3], label="d = "*string(nObj[3]), lw=2, marker=:diamond)
plot!(nVar, cput[1:length(nVar), 4], label="d = "*string(nObj[4]), lw=2, marker=:dtriangle)

plot!(xticks=nVar, xrotation=45)
xlabel!("Number of variables (n)")
ylabel!("Elapsed time (sec)")
title!("Evolution of elapsed time for n and d")
savefig("viewTimeExp4.png")

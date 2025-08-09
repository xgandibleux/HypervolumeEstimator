#= EXPERIMENT 4: 
  - for 100≤n≤500 and 2≤d≤5 given 
    - generate 1 instances ramdomly 
  - for each instance 
    - estimate H for a set of 2000 weights
    - get elapsed times 
  - report average value of
    - elapsed time for H estimated 
=#

using Printf
using Random
       
using JuMP, GLPK #, HiGHS, Gurobi, CPLEX   # for solving MILP
import MultiObjectiveAlgorithms as MOA   # for computing the set of nondominated points
using Distributions                      # for computing the weights and CI (home version)
using SpecialFunctions                   # for computing the estimation value


Random.seed!(1234)

include("src/instanceMO01UKP.jl")
include("src/solveMO01UKP.jl")
include("src/files.jl")
include("src/estimHyperVol1.jl")

println("-"^80)


# =============================================================================
solver = GLPK.Optimizer
#solver = HiGHS.Optimizer
#solver = Gurobi.Optimizer
#solver = CPLEX.Optimizer


listrndWeights = [(2000,2000)]  # number of weights


allCPUt = Matrix{Float64}(undef,10,6)

for n in 100:100:500, o in 2:5

    rp = zeros(Int,o)

    # =============================================================================
    println("  number of variables  : ", n)
    println("  number of objectives : ", o)
    #println("  reference point      : ", rp)
    #println("  interval of #weights : ", listrndWeights)
    println("  solver MIP invoked   : ", solver, "\n")

    # =============================================================================
    p, w, c = generate_MO01UKP(n,o)

    # =============================================================================
    print("  Compute H̃ : ")
    rndWeights = listrndWeights[1]

    startH = time()
    H̃, numberOfWeights = H(solver, p,w,c, rp, rndWeights)
    t_elapsedH = round(time() - startH, digits=2)

    print(" H estimated with rp=$rp and $numberOfWeights weight: ")
    @printf(" %.1f ", round(H̃, digits=2) )
    println(" ($t_elapsedH s)\n")


    allCPUt[Int(n/100),o] = t_elapsedH
end


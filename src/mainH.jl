using Printf
using JuMP, GLPK, Gurobi # for solving MILP
using Distributions      # for computing the weights and CI (home version)
using SpecialFunctions   # for computing the estimation value

include("instanceMO01UKP.jl")
include("estimHyperVol1.jl")


p, w, c = didactic_MO01UKP()

println("Setup the parameters...")
o,n = size(p)    
solver = GLPK.Optimizer
rp = [40,40] # or rp = zeros(Int,o)
rndWeights = (1000,1000)
println("  number of variables  : ", n)
println("  number of objectives : ", o)
println("  reference point      : ", rp)
println("  interval of #weights : ", rndWeights)
println("  solver MIP invoked   : ", solver)

start = time()
H̃, numberOfWeights = H(solver, p,w,c, rp, rndWeights)
t_elapsed = time() - start

println("H estimated with rp=$rp and $numberOfWeights weight: ", round(H̃, digits=2), " ($(round(t_elapsed, digits=2))s)")
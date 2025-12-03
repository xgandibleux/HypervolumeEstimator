#= EXPERIMENT 4: 
  - with 100≤n≤5000 and 2≤d≤10 do 
    - for `n` and `d` given 
    - generate `trial` instances ramdomly  
  - for each instance 
    - estimate H for a set of 2000 weights
    - get elapsed times 
  - report average value of
    - elapsed time for H estimated for n and d given  
=#

using Printf
using Random
       
#using JuMP, GLPK #, HiGHS, Gurobi, CPLEX   # for solving MILP
using JuMP, Gurobi
import MultiObjectiveAlgorithms as MOA   # for computing the set of nondominated points
using Distributions                      # for computing the weights and CI (home version)
using SpecialFunctions                   # for computing the estimation value


Random.seed!(1234)

include("src/instanceMO01UKP.jl")
include("src/solveMO01UKP.jl")
include("src/files.jl")
include("src/estimHyperVol1.jl")
include("src/analyze.jl")


# =============================================================================
#solver = GLPK.Optimizer
#solver = HiGHS.Optimizer
solver = Gurobi.Optimizer
#solver = CPLEX.Optimizer


listrndWeights = [(2000,2000)]  # number of weights
trials = 5 # number of instances generated


allCPUt = Matrix{Float64}(undef,11,11)
global nLines = 1

nVar = [100, 250, 500, 1000, 2500, 5000]
nObj = [2, 3, 5, 10]
 
for n in nVar

  global nCol = 1
  for o in nObj 

    rp = zeros(Int,o)

    # =============================================================================
    println("-"^80)
    println("  number of variables  : ", n)
    println("  number of objectives : ", o)
    #println("  reference point      : ", rp)
    #println("  interval of #weights : ", listrndWeights)
    println("  solver MIP invoked   : ", solver, "\n")

    listCPUt = (Float64)[]

    for _ in 1:trials

      # =============================================================================
      p, w, c = generate_MO01UKP(n,o)

      # =============================================================================
      print("    Compute H̃ : ")
      rndWeights = listrndWeights[1]

      startH = time()
      H̃, numberOfWeights = H(solver, p,w,c, rp, rndWeights)
      t_elapsedH = round(time() - startH, digits=2)

      print("    H estimated with rp=$rp and $numberOfWeights weight: ")
      @printf(" %1.6e ", round(H̃, digits=2) )
      println(" ($t_elapsedH s)\n")
      push!(listCPUt, t_elapsedH)

    end

    avCPUt = average_value(listCPUt)
    @printf("  average CPUt for H̃           = %.2f s\n", avCPUt) 

    allCPUt[nLines,nCol] = avCPUt
    nCol +=1 
  end
  global nLines+=1
end


using Plots

cput = deepcopy(allCPUt)

plot(nVar,  cput[1:length(nVar),1], label = "d = "*string(nObj[1]), lw=2, marker=:circle)
plot!(nVar, cput[1:length(nVar),2], label = "d = "*string(nObj[2]), lw=2, marker=:square)
plot!(nVar, cput[1:length(nVar),3], label = "d = "*string(nObj[3]), lw=2, marker=:diamond)
plot!(nVar, cput[1:length(nVar),4], label = "d = "*string(nObj[4]), lw=2, marker=:dtriangle)
#=
plot!(n, cput[:,5], label = "d = 6", lw=2, marker=:utriangle)
plot!(n, cput[:,6], label = "d = 7", lw=2, marker=:pentagon)
plot!(n, cput[:,7], label = "d = 8", lw=2, marker=:cross)
plot!(n, cput[:,8], label = "d = 9", lw=2, marker=:xcross)
plot!(n, cput[:,9], label = "d = 10", lw=2, marker=:star)
=#

# Personnalisation
plot!(xticks=nVar,xrotation = 45)
xlabel!("Number of variables (n)")
ylabel!("Elapsed time (sec)")
title!("Evolution of elapsed time for n and d")
#grid!(true)
savefig("viewTimeExp4.png")

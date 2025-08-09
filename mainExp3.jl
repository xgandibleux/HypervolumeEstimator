#= EXPERIMENT 3: 
  - for n and d given 
    - generate 20 instances ramdomly 
  - for each instance 
    - compute Y_N
    - measure H 
    - estimate H for a set of 2000 weights
    - compute relative error 
    - get elapsed times 
  - report average value of
    - elapsed time for H measured
    - elapsed time for H estimated
    - relative error     
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
println("Setup the parameters...")
solver = GLPK.Optimizer
#solver = HiGHS.Optimizer
#solver = Gurobi.Optimizer
#solver = CPLEX.Optimizer

n = 25            # number of variables
o = 5             # number of objectives
nInstances = 20   # number of nInstances

rp = zeros(Int,o)
listrndWeights = [(2000,2000)]  # number of weights

listH     = (Float64)[]
listH̃     = (Float64)[]
listCPUtH = (Float64)[]
listCPUtH̃ = (Float64)[]
listabsolue_error  = (Float64)[]
listrelative_error = (Float64)[]


# =============================================================================
println("  number of variables  : ", n)
println("  number of objectives : ", o)
println("  reference point      : ", rp)
println("  interval of #weights : ", listrndWeights)
println("  solver MIP invoked   : ", solver)
println("  number of nInstances : ", nInstances)

for iInstance in 1:nInstances

    # =============================================================================
    println("\nInstance $iInstance ...")
    p, w, c = generate_MO01UKP(n,o)

    # =============================================================================
    print("\n  Compute S : ")
    start = time()
    S, cardS = solve_MO01UKP(solver, p, w, c)
    t_elapsedS = round(time() - start, digits=2)
    println(" |S|  = ",cardS, " ($t_elapsedS s)")

    # =============================================================================
    print("  Compute H : ")
    writeOnFile_S("HVpoints", S)
    if o == 2
        run(pipeline(`./src/hv -r "0 0" HVpoints`, stdout="HVmeasure"))
    elseif o == 3
        run(pipeline(`./src/hv -r "0 0 0" HVpoints`, stdout="HVmeasure"))
    elseif o == 4
        run(pipeline(`./src/hv -r "0 0 0 0" HVpoints`, stdout="HVmeasure"))
    elseif o == 5
        run(pipeline(`./src/hv -r "0 0 0 0 0" HVpoints`, stdout="HVmeasure"))
    elseif o == 6
        run(pipeline(`./src/hv -r "0 0 0 0 0 0" HVpoints`, stdout="HVmeasure"))
    end
    Hmeasure = read_Hmeasure("HVmeasure")
    @printf(" H(S) = %.1f\n", Hmeasure)

    push!(listH, Hmeasure)
    push!(listCPUtH, t_elapsedS)


    # =============================================================================
    print("  Compute H̃ : ")
    rndWeights = listrndWeights[1]

    startH = time()
    H̃, numberOfWeights = H(solver, p,w,c, rp, rndWeights)
    t_elapsedH = round(time() - startH, digits=2)

    print(" H estimated with rp=$rp and $numberOfWeights weight: ")
    @printf(" %.1f ", round(H̃, digits=2) )
    println(" ($t_elapsedH s)")

    push!(listH̃, H̃)
    push!(listCPUtH̃, t_elapsedH)

    # =============================================================================
    absolue_error = abs(Hmeasure - H̃)
    relative_error = abs((Hmeasure - H̃) / Hmeasure)

    push!(listabsolue_error, absolue_error)
    push!(listrelative_error, relative_error)

end


# =============================================================================
println("\nSummary...\n")

@printf("  average absolue error H̃     = %.1f \n", mean(listabsolue_error))
@printf("  average relative error H̃    = %.6f \n", mean(listrelative_error))
@printf("  average CPUt for S          = %.2f s\n", mean(listCPUtH)) 
@printf("  average CPUt for H̃          = %.2f s\n", mean(listCPUtH̃))   
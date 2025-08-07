using Printf
using Random
       
using JuMP, GLPK #, HiGHS, Gurobi, CPLEX   # for solving MILP
import MultiObjectiveAlgorithms as MOA   # for computing the set of nondominated points
using Distributions                      # for computing the weights and CI (home version)
using SpecialFunctions                   # for computing the estimation value
using HypothesisTests                    # for computing the confidence interval (package version)
using Statistics                         # for computing the confidence interval (home version)
using Plots                              # for drawing the figure (evolution of the avg relative error)

Random.seed!(1234)

include("src/instanceMO01UKP.jl")
include("src/solveMO01UKP.jl")
include("src/files.jl")
include("src/estimHyperVol1.jl")
include("src/analyze.jl")
include("src/computeCI.jl")

println("-"^80)

mutable struct resultsExpe
    x::Vector{Int64}
    Hmeasure::Float64
    avH̃::Vector{Float64}
    CIhight::Vector{Float64}
    CIlow::Vector{Float64}
end

#oneExpe = resultsExpe(  [100,500,1000,1500,2000,5000,10000], 
#                        0.0, 
#                        zeros(7),
#                        zeros(7),
#                        zeros(7)
#                    )

# =============================================================================
println("Setup the parameters...")
solver = GLPK.Optimizer
#solver = HiGHS.Optimizer
#solver = Gurobi.Optimizer
#solver = CPLEX.Optimizer
n = 50    # number of variables
o = 3     # number of objectives

rp = zeros(Int,o)
listrndWeights = [(2000,2000)]  # number of weights
nInstances     = 20              # number of nInstances

println("  number of variables  : ", n)
println("  number of objectives : ", o)
println("  reference point      : ", rp)
println("  interval of #weights : ", listrndWeights)
println("  solver MIP invoked   : ", solver)
println("  number of nInstances : ", nInstances)

#allareH̃ = (Float64)[]
#allaCPUt = (Float64)[]

    #open(instanceName*".res", "w") do ioAll
    #    write(ioAll, string(instanceName,"\n"))

listH     = (Float64)[]
listH̃     = (Float64)[]
listCPUtH = (Float64)[]
listCPUtH̃ = (Float64)[]
listabsolue_error  = (Float64)[]
listrelative_error = (Float64)[]

for iInstance in 1:nInstances
    instanceName = "kp-" * string(n) * "-" * string(o) * "-" * string(iInstance)

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


    #    write(ioAll, string("H(S) = ",Hmeasure, " \n"))
    #    write(ioAll, string("\n"))


    # =============================================================================

    print("  Compute H̃ : ")
    rndWeights = listrndWeights[1]

    startH = time()
    H̃, numberOfWeights = H(solver, p,w,c, rp, rndWeights)
    t_elapsedH = round(time() - startH, digits=2)

    print(" H estimated with rp=$rp and $numberOfWeights weight: ")
    @printf(" %.1f ", round(H̃, digits=2) )
    println(" ($t_elapsedH s)")

#   write(ioAll, string(numberOfWeights, " ", round(H̃, digits=2), " ",t_elapsedH, "s\n"))

    push!(listH̃, H̃)
    push!(listCPUtH̃, t_elapsedH)

    # =============================================================================
    absolue_error = abs(Hmeasure - H̃)
    relative_error = abs((Hmeasure - H̃) / Hmeasure)

    push!(listabsolue_error, absolue_error)
    push!(listrelative_error, relative_error)

#   write(ioAll, string("\n"))

end


# -------------------------------------------------------------------------
println("\nSummary...\n")

@printf("  average absolue error H̃     = %.1f \n", mean(listabsolue_error))
@printf("  average relative error H̃    = %.6f \n", mean(listrelative_error))
@printf("  average CPUt for S          = %.2f s\n", mean(listCPUtH)) 
@printf("  average CPUt for H̃          = %.2f s\n", mean(listCPUtH̃))   

     



#println("\nAll average relative error H̃ = ", allareH̃)
#println("\nAll CPUt with ", solver, " = ", allaCPUt)

#=
listrndWeightsX = [100,500,1000,1500,2000,5000,10000]
plot(listrndWeightsX, allareH̃, 
    seriestype = :line, 
    marker = :circle,
    title = string(n)*" variables | "*string(o)*" objectives | "*string(nInstances)*" nInstances",
    xlabel = "Number of weights",
    ylabel = "average relative error",
    legend = false,
    linewidth = 2,
    xticks = listrndWeightsX,
    xrotation = 45,
    show = true
)

savefig("kp-"*string(n)*"-"*string(o))
nothing


function plot_values(oneExpe::resultsExpe)

    exact = fill(oneExpe.Hmeasure,7)

    yerr_low = oneExpe.avH̃ .- oneExpe.CIlow
    yerr_high = oneExpe.CIhight .- oneExpe.avH̃ 

    plot(listrndWeightsX, oneExpe.avH̃, yerror = (yerr_low, yerr_high),
         label = "Avg Estimated ± CI", lw=2, marker=:circle, color=:red,
         xticks = (listrndWeightsX, string.(listrndWeightsX)), xrotation = 45
    )

    plot!(listrndWeightsX, exact, label = "Exact", marker=:diamond, ms=6, color=:black, linestyle=:dash)

    xlabel!("Number of weights")
    ylabel!("Hypervolume value")
    title!(string(n)*" variables | "*string(o)*" objectives | "*string(nInstances)*" nInstances | CI 95%")
end

plot_values(oneExpe)
savefig("H"*string(n)*"-"*string(o))
=#
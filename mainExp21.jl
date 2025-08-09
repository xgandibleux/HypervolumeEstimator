#= EXPERIMENT 2.1: 
  - for n and d given 
    - generate 1 instance ramdomly 
  - for each instance 
    - compute Y_R
    - measure H 
    - estimate H for 7 sets of weights, each set is composed of 100,500,1000,1500,2000,5000,10000 weights
    - compute relative error 
    - collect elapsed times 
  - report for each instance 
    - cardinality of Y_R
    - elapsed time for computing Y_R   
    - H measured     
    - for each set of weights    
      - average value of H estimated
      - average value of elapsed time for H estimated
      - confidence interval for 95%
      - average value of relative error     
=#

using Printf
using Random
       
using JuMP, GLPK                         # for solving MILP (I)
#using HiGHS, Gurobi, CPLEX              # for solving MILP (II)
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

oneExpe = resultsExpe(  [100,500,1000,1500,2000,5000,10000], 
                        0.0, 
                        zeros(7),
                        zeros(7),
                        zeros(7)
                    )

# =============================================================================
println("Setup the parameters...")
solver = GLPK.Optimizer
#solver = HiGHS.Optimizer
#solver = Gurobi.Optimizer
#solver = CPLEX.Optimizer
n = 200    # number of variables
o = 3     # number of objectives
nWeights = n*o   # number of weights for the scalarizing function


rp = zeros(Int,o)
listrndWeights = [(100,100), (500,500), (1000,1000), (1500,1500), (2000,2000), (5000,5000), (10000,10000)]
trials = 20 # number of trials

println("  number of variables  : ", n)
println("  number of objectives : ", o)
println("  reference point      : ", rp)
println("  interval of #weights : ", listrndWeights)
println("  solver MIP invoked   : ", solver)
println("  number of trials     : ", trials)

allareH̃ = (Float64)[]
allaCPUt = (Float64)[]

function evaluateSolution(p, x)
    o,n = size(p)
    z = zeros(Int,o)

    for k in 1:o, j in 1:n
        z[k] += p[k,j] * x[j]
    end

    return z
end

instanceName = "kp-" * string(n) * "-" * string(o)
open(instanceName*".res", "w") do ioAll
    write(ioAll, string(instanceName,"\n"))


    # =============================================================================
    println("\nGenerate an mo01UKP instance...")
    p, w, c = generate_MO01UKP(n,o)
    save_instance(instanceName* ".dat", p, w, c)

    #S, cardS = solve_MO01UKP(solver, p, w, c)
    #@show S, cardS

    # =============================================================================
    println("\nCompute R, a set of nondominated points...")
    start = time()
    
    # -------------------------------------------------------------------------
    print("\n  1) Compute zᴵ = ")
    zIdeal = Array{Int}(undef,o)
    zUtopian = Array{Int}(undef,o)
 
    start = time()
    for k in 1:o
        zIdeal[k], xOpt = solve_01UKP(solver, p[k,:], w, c)
    end
    zUtopian = Int.(ceil.(zIdeal .* 1.01))
    t_elapsedU = round(time() - start, digits=2)
    println(zIdeal, "  zᵁ = ", zUtopian, "  ($t_elapsedU s)")

   # -------------------------------------------------------------------------
    R = Set{Vector{Int64}}([])
    for _ in 1:nWeights

        # compute a weight 
        rnd = rand(1:100, o)
        λ = rnd ./ sum(rnd)

        # Compute a nondominated points using the weighted Tchebychev norm
        zOpt, xOpt = solve_scalarized01UKP(solver, p, w, c, zUtopian, λ)

        # Evaluate one solution on the objective functions
        z = evaluateSolution(p, xOpt)

        if z ∉ R
            push!(R,z)
        end
    end
    #@show R

    t_elapsedR = round(time() - start, digits=2) 

    save_nondominatedpoints(instanceName*".yr",R)
    println("  |R|  = ",length(R), " ($t_elapsedR)s)")
    write(ioAll, string("|R|  = ",length(R), " ($t_elapsedR s) \n"))
    #write(ioAll, string("\n"))

    #for z in R
    #    if z ∉ S
    #        @assert false "WND point found " * string(z)
    #    end
    #end

    #@assert false "stop"

    # =============================================================================
    println("\nCompute H, the hypervolume measure...")
    writeOnFile_S("HVpoints", R)
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
    @printf(". H(S) = %.1f\n", Hmeasure)
    oneExpe.Hmeasure = Hmeasure

    write(ioAll, string("H(S) = ",Hmeasure, " \n"))
    write(ioAll, string("\n"))


    # reset the random generator
    Random.seed!(1234)

    # =============================================================================
    for iWeight in 1:length(listrndWeights)
        rndWeights = listrndWeights[iWeight]

        # -------------------------------------------------------------------------
        println("\nCompute H̃, the estimation of H...")
        listH̃ = (Float64)[]
        listCPUt = (Float64)[]

        for _ in 1:trials
            startH = time()
            H̃, numberOfWeights = H(solver, p,w,c, rp, rndWeights)
            t_elapsedH = round(time() - startH, digits=2)

            print("  H estimated with rp=$rp and $numberOfWeights weight: ")
            @printf(" %.1f ", round(H̃, digits=2) )
            println(" ($t_elapsedH s)")

            write(ioAll, string(numberOfWeights, " ", round(H̃, digits=2), " ",t_elapsedH, "s\n"))

            push!(listH̃, H̃)
            push!(listCPUt, t_elapsedH)
        end
        write(ioAll, string("\n"))

        # -------------------------------------------------------------------------
        # Confidence_interval with HypothesisTests package
        CIlow, CIHigh = confint( OneSampleTTest( listH̃ ), level=0.95, tail=:both )

        # -------------------------------------------------------------------------
        println("\nAnalyze the results...")
        avH̃ = average_value(listH̃)
        avCPUt = average_value(listCPUt)
        aaeH̃ = average_absolue_error(Hmeasure, listH̃)
        areH̃ = average_relative_error(Hmeasure, listH̃)

        @printf("  value H(S)                  = %.1f \n", Hmeasure)
        @printf("  average value H̃             = %.1f \n", avH̃)
        @printf("  average absolue error H̃     = %.1f \n", aaeH̃)
        @printf("  average relative error H̃    = %.6f \n", areH̃)
        #@printf("  confidence interval for 95%% = [%.1f, %.1f] \n", ci[1], ci[2])
        @printf("  confidence interval for 95%% = [%.1f, %.1f] \n",CIlow, CIHigh)
        @printf("  CPUt for computing S         = %.2f s\n", t_elapsedR)
        @printf("  average CPUt for H̃           = %.2f s\n", avCPUt)    
        push!(allareH̃, areH̃)
        push!(allaCPUt, avCPUt)

        write(ioAll, string("average value H̃             = ",avH̃, " \n"))
        write(ioAll, string("average absolue error H̃     = ",aaeH̃, " \n"))
        write(ioAll, string("average relative error H̃    = ",areH̃, " \n"))
        write(ioAll, string("confidence interval for 95% = ",CIlow, " ", CIHigh, " \n"))
        write(ioAll, string("average CPUt for H̃          = ",avCPUt, " \n\n"))

        oneExpe.avH̃[iWeight] = avH̃
        oneExpe.CIhight[iWeight] = CIHigh
        oneExpe.CIlow[iWeight] = CIlow
    end

end

println("\nAll average relative error H̃ = ", allareH̃)
println("\nAll CPUt with ", solver, " = ", allaCPUt)


listrndWeightsX = [100,500,1000,1500,2000,5000,10000]
plot(listrndWeightsX, allareH̃, 
    seriestype = :line, 
    marker = :circle,
    title = string(n)*" variables | "*string(o)*" objectives | "*string(trials)*" trials",
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

    plot!(listrndWeightsX, exact, label = "Representative", marker=:diamond, ms=6, color=:black, linestyle=:dash)

    xlabel!("Number of weights")
    ylabel!("Hypervolume value")
    title!(string(n)*" variables | "*string(o)*" objectives | "*string(trials)*" trials | CI 95%")
end

plot_values(oneExpe)
savefig("H"*string(n)*"-"*string(o))
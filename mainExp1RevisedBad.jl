#= EXPERIMENT 1 REVISED: 

  - given a value for n and d  
    - generate 1 instance ramdomly 
    - compute Y_N
    - measure H 
    - estimate H for 7 sets of weights, each set is composed of 100,500,1000,1500,2000,5000,10000 weights
    - collect elapsed times 

  - report
    - cardinality of Y_N
    - elapsed time for computing Y_N   
    - H measured        
    - H estimated
    - elapsed time for H estimated
    - confidence interval for 95%
    - absolute error
    - relative error     
=#

using Printf
using Random
       
#using JuMP, GLPK                         # for solving MILP (I)
using JuMP, Gurobi                         # for solving MILP (I)
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
    x        :: Vector{Int64}
    Hmeasure :: Float64
    avL      :: Vector{Float64}
    CIhightL :: Vector{Float64}
    CIlowL   :: Vector{Float64}
end

oneExpe = resultsExpe(  [100,500,1000,1500,2000,5000,10000], 
                        0.0, 
                        zeros(7),
                        zeros(7),
                        zeros(7)
                    )

# =============================================================================
println("Setup the parameters...")
#solver = GLPK.Optimizer
#solver = HiGHS.Optimizer
solver = Gurobi.Optimizer
#solver = CPLEX.Optimizer
n = 100    # number of variables
o = 3      # number of objectives

rp = zeros(Int,o)
listnbrWeights = [100,500,1000,1500,2000,5000,10000] 

println("  number of variables  : ", n)
println("  number of objectives : ", o)
println("  reference point      : ", rp)
println("  number of weights    : ", listnbrWeights)
println("  solver MIP invoked   : ", solver)

allareH̃ = (Float64)[]
allaCPUt = (Float64)[]

instanceName = "kp-" * string(n) * "-" * string(o)
open(instanceName*".res", "w") do ioAll
    write(ioAll, string(instanceName,"\n"))

    # =============================================================================
    println("\nGenerate an mo01UKP instance...")
    p, w, c = generate_MO01UKP(n,o)
    save_instance(instanceName* ".dat", p, w, c)

    # =============================================================================
    println("\nCompute S, the set of nondominated points...")
    start = time()
    S, cardS = solve_MO01UKP(solver, p, w, c)
    t_elapsedS = round(time() - start, digits=2)

    save_nondominatedpoints(instanceName*".yn",S)
    println("  |S|  = ",cardS, " ($t_elapsedS)s)")
    write(ioAll, string("|S|  = ",cardS, " ($t_elapsedS s) \n"))

    # =============================================================================
    println("\nCompute H, the hypervolume measure...")
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
    @printf("  H(S) = %1.6e\n", Hmeasure)    
    oneExpe.Hmeasure = Hmeasure

    write(ioAll, string("H(S) = ",Hmeasure, " \n"))
    write(ioAll, string("\n"))


    # reset the random generator
    Random.seed!(12345)

    # =============================================================================
    for iWeight in 1:length(listnbrWeights)
        nbrWeights = listnbrWeights[iWeight]

        listH̃ = (Float64)[]
        listCPUt = (Float64)[]

        # -------------------------------------------------------------------------
        println("\nCompute H̃, the estimation of H with $nbrWeights weights...")
        startH = time()
        H̃, numberOfWeights, listL = Hrevised(solver, p,w,c, rp, nbrWeights)
        t_elapsedH = round(time() - startH, digits=2)

        write(ioAll, string(numberOfWeights, " ", round(H̃, digits=2), " ",t_elapsedH, "s\n"))
        push!(listH̃, H̃)
        push!(listCPUt, t_elapsedH)
        write(ioAll, string("\n"))


        # -------------------------------------------------------------------------
        println("\nResults...")

        # REVISION -------------------------------------------------------------------------
        # Confidence_interval with HypothesisTests package
        CIlowL, CIHighL = confint( OneSampleTTest( listL ), level=0.95, tail=:both )
        avL = average_value(listL)
        # REVISION -------------------------------------------------------------------------

        avCPUt = average_value(listCPUt)
        aaeH̃ = average_absolue_error(Hmeasure, listH̃)
        areH̃ = average_relative_error(Hmeasure, listH̃)

        @printf("  value H(S)                  = %1.6e \n", Hmeasure)
        @printf("  value H̃                     = %1.6e \n", H̃)
        @printf("  average value L             = %1.6e \n", avL)
        @printf("  confidence interval for 95%% = [%1.6e, %1.6e] \n",CIlowL, CIHighL)        
        @printf("  absolue error H̃             = %1.6e \n", aaeH̃)
        @printf("  relative error H̃            = %.6f \n", areH̃)
        @printf("  CPUt for computing S        = %.2f s\n", t_elapsedS)
        @printf("  CPUt for computing H̃        = %.2f s\n", t_elapsedH)    

        push!(allareH̃, areH̃)
        push!(allaCPUt, avCPUt)

        write(ioAll, string("value H̃                     = ", H̃, " \n"))
        write(ioAll, string("absolue error H̃             = ", aaeH̃, " \n"))
        write(ioAll, string("relative error H̃            = ", areH̃, " \n"))
        write(ioAll, string("confidence interval for 95% = ", CIlowL, " ", CIHighL, " \n"))
        write(ioAll, string("CPUt for H̃                  = ", avCPUt, " \n\n"))

        oneExpe.avL[iWeight] = avL
        oneExpe.CIhightL[iWeight] = CIHighL
        oneExpe.CIlowL[iWeight] = CIlowL
    end
end

println("\nAll average relative error H̃ = ", allareH̃)
println("\nAll CPUt with ", solver, " = ", allaCPUt)


plot(listnbrWeights, allareH̃, 
    seriestype = :line, 
    marker = :circle,
    title = string(n)*" variables | "*string(o)*" objectives", 
    xlabel = "Number of weight vectors/iterations",
    ylabel = "Relative error",
    legend = false,
    linewidth = 2,
    xticks = listnbrWeights,
    xrotation = 45,
    show = true
)

savefig("kp-"*string(n)*"-"*string(o))
nothing


function plot_values(oneExpe::resultsExpe)

    yerr_low = oneExpe.avL .- oneExpe.CIlowL
    yerr_high = oneExpe.CIhightL .- oneExpe.avL 

    plot(listnbrWeights, oneExpe.avL, yerror = (yerr_low, yerr_high),
         label = "Avg Estimated ± CI", lw=2, marker=:circle, color=:red,
         xticks = (listnbrWeights, string.(listnbrWeights)), xrotation = 45
    )

    xlabel!("Number of weight vectors/iterations")
    ylabel!("Optimal values of Chebychev models")
    title!(string(n)*" variables | "*string(o)*" objectives | CI 95%")
end

plot_values(oneExpe)
savefig("H"*string(n)*"-"*string(o))
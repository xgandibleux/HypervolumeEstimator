#= EXPERIMENT 1 REVISED (2): 

  - given a value for n and d  
    - generate 1 instance ramdomly 
    - compute Y_N
    - measure H 
    - repeat trials times 
      - estimate H for 7 sets of weights, each set is composed of 100,500,1000,1500,2000,5000,10000 weights
      - collect elapsed times 
    - compute 
      - average value of H estimated
      - confidence interval for 95%
      - absolute error
      - relative error  

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
    x          :: Vector{Int64}
    Hmeasure   :: Float64
    avH̃        :: Vector{Float64}
    CIhight    :: Vector{Float64}
    CIlow      :: Vector{Float64}
    avLw       :: Vector{Float64}
    CIhightLw  :: Vector{Float64}
    CIlowLw    :: Vector{Float64}    
end



# =============================================================================
println("Setup the parameters...")
#solver = GLPK.Optimizer
#solver = HiGHS.Optimizer
solver = Gurobi.Optimizer
#solver = CPLEX.Optimizer

# number of variables
n = 25    
# number of objectives
o = 5     

# reference point used for the H measure
rp = zeros(Int,o)
# list of number of weights to use
listnbrWeights = [100,500,1000,1500,2000,5000,10000] 
#listnbrWeights = [1000,1000,1000,1000,1000,1000,1000] 
# number of trials used for the experiment
trials = 10 
# number of instances in the experiment 1 or 7
nInstances = 1

# reset the random generator
Random.seed!(1234)

println("  number of variables  : ", n)
println("  number of objectives : ", o)
println("  reference point      : ", rp)
println("  number of weights    : ", listnbrWeights)
println("  solver MIP invoked   : ", solver)
println("  number of trials     : ", trials)

allareH̃ = (Float64)[]    # all average_relative_error on estimation of H 
allaCPUt = (Float64)[]   # all average_value of CPUt 

#allavgCIlower = (Float64)[]   # all average CI lower 
#allavgCIupper = (Float64)[]   # all average CI upper 
W = 0.0                       # average precision

oneExpe = resultsExpe(  listnbrWeights, #[100,500,1000,1500,2000,5000,10000], 
                        0.0, 
                        zeros(7),
                        zeros(7),
                        zeros(7),
                        zeros(7),
                        zeros(7),
                        zeros(7)                        
                    )


#
# File "tableResultsExpe1.txt" collecting all results for a table in the paper
#
fresults = open("tableResultsExpe1.res", "a")

# 
# File collecting all results for one instance
#
instanceName = "kp-" * string(n) * "-" * string(o)
open(instanceName*".res", "w") do ioAll
    write(ioAll, string(instanceName,"\n"))

    for instanceID in 1:nInstances

        # ==== PART 0: INSTANCE ===================================================

        # -------------------------------------------------------------------------
        println("\nGenerate an mo01UKP instance...")
        p, w, c = generate_MO01UKP(n,o)
        save_instance(instanceName* ".dat", p, w, c)


        # ==== PART 1: EXACT ======================================================

        # -------------------------------------------------------------------------
        println("\nCompute S = Y_N, the set of nondominated points...")
        start = time()
        S, cardS = solve_MO01UKP(solver, p, w, c)
        save_nondominatedpoints(instanceName*".yn",S)

        t_elapsedS = round(time() - start, digits=2)
        println("  |S|  = ",cardS, " ($t_elapsedS)s)")
        write(ioAll, string("|S|  = ",cardS, " ($t_elapsedS s) \n"))

        # -------------------------------------------------------------------------
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

        # ==== Part 2: ESTIMATION =====================================================

        # =============================================================================
        for iWeight in 1:length(listnbrWeights)
            nbrWeights = listnbrWeights[iWeight]

            # -------------------------------------------------------------------------
            println("\nCompute H̃, the estimation of H...\n")
            listH̃ = (Float64)[]
            listCPUt = (Float64)[]

            list_meanCI = (Float64)[]
            list_lowerCI = (Float64)[]
            list_upperCI = (Float64)[]

            for _ in 1:trials
                startH = time()
                H̃, numberOfWeights, info_for_CI = Hrevised2(solver, p,w,c, rp, nbrWeights)
                t_elapsedH = round(time() - startH, digits=2)

                print("  H estimated with rp=$rp  and  $numberOfWeights weight: ")
                @printf(" %.1f ", round(H̃, digits=2) )
                println(" ($t_elapsedH s)")

                print("  average value L weighted = %1.6e  -->>  ")
                @printf(" %.1f ", round(info_for_CI[1], digits=2) )
                print("confidence interval for 95%% = ") 
                @printf(" [%.1f, %.1f] \n", round(info_for_CI[2], digits=2), round(info_for_CI[3], digits=2) )
                push!(list_meanCI, info_for_CI[1])
                push!(list_lowerCI, info_for_CI[2])
                push!(list_upperCI, info_for_CI[3])

                write(ioAll, string(numberOfWeights, " ", round(H̃, digits=2), " ",t_elapsedH, "s\n"))

                push!(listH̃, H̃)
                push!(listCPUt, t_elapsedH)

            end # several trials

            write(ioAll, string("\n"))

            #mu, ci_low, ci_high = combine_ci(list_meanCI, list_lowerCI, list_upperCI)
            #print("\n  Combined estimate: ")
            #@printf("L weighted = %1.6e  -->>  ", mu)
            #@printf("95%% CI = [%1.6e, %1.6e] \n",ci_low, ci_high)         
            # -------------------------------------------------------------------------
            # Confidence_interval with HypothesisTests package
            #CIlow, CIHigh = confint( OneSampleTTest( listH̃ ), level=0.95, tail=:both )

            # -------------------------------------------------------------------------
            println("\nAnalyze the results...")
            avH̃ = average_value(listH̃)
            avCPUt = average_value(listCPUt)
            aaeH̃ = average_absolue_error(Hmeasure, listH̃)
            areH̃ = average_relative_error(Hmeasure, listH̃)

            @printf("  value H(S)                  = %1.6e \n", Hmeasure)
            @printf("  average value H̃             = %1.6e \n", avH̃)
            #@printf("  average absolue error H̃     = %1.6e \n", aaeH̃)
            #@printf("  average relative error H̃    = %.6f \n", areH̃)
        #@printf("  confidence interval for 95%% = [%1.6e, %1.6e] \n",CIlow, CIHigh)
            @printf("  CPUt for computing S         = %.2f s\n", t_elapsedS)
            @printf("  average CPUt for H̃           = %.2f s\n", avCPUt)    
            #push!(allareH̃, areH̃)
            push!(allaCPUt, avCPUt)

            write(ioAll, string("average value H̃             = ",avH̃, " \n"))
            #write(ioAll, string("average absolue error H̃     = ",aaeH̃, " \n"))
            #write(ioAll, string("average relative error H̃    = ",areH̃, " \n"))
            #write(ioAll, string("confidence interval for 95% = ",CIlow, " ", CIHigh, " \n"))
            write(ioAll, string("average CPUt for H̃          = ",avCPUt, " \n\n"))

            oneExpe.avH̃[iWeight]       = avH̃
            oneExpe.CIhight[iWeight]   = 0.0 #CIHigh
            oneExpe.CIlow[iWeight]     = 0.0 #CIlow

            oneExpe.avLw[iWeight]      = average_value(list_meanCI)  # average value of average L_weighted
            oneExpe.CIlowLw[iWeight]   = average_value(list_lowerCI) # average value of lower CI
            oneExpe.CIhightLw[iWeight] = average_value(list_upperCI) # average value of upper CI
            
            @printf("  average value of upper CI           = %1.6e \n", oneExpe.CIhightLw[iWeight])
            @printf("  average value of average L_weighted = %1.6e \n", oneExpe.avLw[iWeight])        
            @printf("  average value of lower CI           = %1.6e \n", oneExpe.CIlowLw[iWeight])           
            
            avg_relative_error = abs( (Hmeasure - avH̃) / Hmeasure)   #  |exact hv - our estimated hv| / exact hv
            push!(allareH̃, avg_relative_error)

            println("\nWriting results on file...")
            print(fresults," $n  &  $o  &  $cardS  & ")
            @printf(fresults," %.2f  & ", t_elapsedS)
            @printf(fresults," %1.6e  & ", Hmeasure)
            print(fresults," $nbrWeights  & ")
            @printf(fresults," %1.6e  & ", avH̃) 
            @printf(fresults," %.2f  & ",  avCPUt)
            @printf(fresults," %1.6e  & ", oneExpe.CIlowLw[iWeight])
            @printf(fresults," %1.6e  & ", oneExpe.CIhightLw[iWeight])
            @printf(fresults," %.6f \n", avg_relative_error)
            #println(f, "Contenu")

        end # several weights

    end # several instances

end

# closing the file collecting all results
close(fresults)


println("\nAll average relative error H̃ = ", allareH̃)
println("All CPUt with ", solver, " = ", allaCPUt)

println("\nAll average value of upper CI           = ", oneExpe.CIhightLw)
println("All average value of average L_weighted = ", oneExpe.avLw)
println("All average value of upper CI           = ", oneExpe.CIlowLw)


# ==== PLOTS ==================================================================

#listnbrWeights = [100,500,1000,1500,2000,5000,10000]
plot(listnbrWeights, allareH̃, 
    seriestype = :line, 
    marker = :circle,
    title = string(n)*" variables | "*string(o)*" objectives | "*string(trials)*" trials",
    xlabel = "Number of weight vectors/iterations",
    ylabel = "average relative error",
    legend = false,
    linewidth = 2,
    xticks = listnbrWeights,
    xrotation = 45,
    show = true
)

savefig("kp-"*string(n)*"-"*string(o))
nothing


function plot_values(oneExpe::resultsExpe)

    exact = fill(oneExpe.Hmeasure,7)

    yerr_low = oneExpe.avH̃ .- oneExpe.CIlow
    yerr_high = oneExpe.CIhight .- oneExpe.avH̃ 

    plot(listnbrWeights, oneExpe.avH̃, yerror = (yerr_low, yerr_high),
         label = "Avg Estimated ± CI", lw=2, marker=:circle, color=:red,
         xticks = (listnbrWeights, string.(listnbrWeights)), xrotation = 45
    )

    plot!(listnbrWeights, exact, label = "Exact", marker=:diamond, ms=6, color=:black, linestyle=:dash)

    xlabel!("Number of weight vectors/iterations")
    ylabel!("Hypervolume value")
    title!(string(n)*" variables | "*string(o)*" objectives | "*string(trials)*" trials | CI 95%")
end

function plot_valuesCI2(oneExpe::resultsExpe)

    exact = fill(oneExpe.Hmeasure,7)

    yerr_low = oneExpe.avLw .- oneExpe.CIlowLw
    yerr_high = oneExpe.CIhightLw .- oneExpe.avLw 

    plot(listnbrWeights, oneExpe.avLw, yerror = (yerr_low, yerr_high),
         label = "Average ± CI", lw=2, marker=:circle, color=:red,
         xticks = (listnbrWeights, string.(listnbrWeights)), xrotation = 45
    )

    plot!(listnbrWeights, exact, label = "Exact", marker=:diamond, ms=6, color=:black, linestyle=:dash)

    xlabel!("Number of weight vectors/iterations")
    ylabel!("L weighted value")
    title!(string(n)*" variables | "*string(o)*" objectives | "*string(trials)*" trials | CI 95%")
end

#plot_values(oneExpe)
#savefig("H"*string(n)*"-"*string(o))

plot_valuesCI2(oneExpe)
savefig("Lw"*string(n)*"-"*string(o))
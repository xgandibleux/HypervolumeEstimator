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
#using LaTeXStrings                       # for using latex commands in Plots legends

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


function plot_valuesCI2(oneExpe7Weights::resultsExpe)

#    exact = fill(oneExpe7Weights.Hmeasure,7)
    exact = fill(1,7)   

    yerr_low = oneExpe7Weights.avLw .- oneExpe7Weights.CIlowLw
    yerr_high = oneExpe7Weights.CIhightLw .- oneExpe7Weights.avLw 

    plot(listnbrWeights, oneExpe7Weights.avLw, yerror = (yerr_low, yerr_high),
        label = "Average ± CI", lw=2, marker=:circle, color=:red,
        xticks = (listnbrWeights, string.(listnbrWeights)), xrotation = 45
    )

    plot!(listnbrWeights, exact, label = "Exact", marker=:diamond, ms=6, color=:black, linestyle=:dash)

    xlabel!("Number of weight vectors/iterations")
    ylabel!("H estimated normalized")
    title!(string(n)*" variables | "*string(o)*" objectives | CI 95%")
end


function plot_valuesCI3(oneExpe7Weights::resultsExpe, res_H_estimated_normalized)
    exact = fill(1, 7)
    yerr_low  = oneExpe7Weights.avLw .- oneExpe7Weights.CIlowLw
    yerr_high = oneExpe7Weights.CIhightLw .- oneExpe7Weights.avLw

    plot(listnbrWeights, exact, label = "Exact H", marker=:diamond, ms=6, color=:black, linestyle=:dash)

    # Ajout des valeurs individuelles colonne par colonne
    n_rows = size(res_H_estimated_normalized, 1)
    x_scattered = repeat(listnbrWeights, inner = n_rows)  # [x1,x1,...,x2,x2,...,x7,x7,...]
    y_scattered = vec(res_H_estimated_normalized)          # colonne 1, puis 2, ... puis 7

    #scatter!(x_scattered, y_scattered,
    #    label = "Individual runs", color=:green, alpha=0.4, ms=3, markerstrokewidth=0
    #)

    scatter!(x_scattered, y_scattered,
    label = "Individual H estimated", color=:blue, alpha=0.75, marker=:hline, ms=6, markerstrokewidth=1
    )

    plot!(listnbrWeights, oneExpe7Weights.avLw, yerror = (yerr_low, yerr_high),
    label = "Average H estimated ± average CI", lw=2, marker=:circle, color=:red,
    xticks = (listnbrWeights, string.(listnbrWeights)), xrotation = 45
    )

    xlabel!("Number of weight vectors/iterations")
    ylabel!("H estimated normalized")
    title!(string(n)*" variables | "*string(o)*" objectives | CI 95%")
end

# =============================================================================
println("Setup the parameters...")
#solver = GLPK.Optimizer
#solver = HiGHS.Optimizer
solver = Gurobi.Optimizer
#solver = CPLEX.Optimizer


# number of variables
n = 10   
# number of objectives
o = 3     

# reference point used for the H measure
rp = zeros(Int,o)


# number of instances 
nInstances = 10
# list of number of weights to use
listnbrWeights = [100,500,1000,1500,2000,5000,10000]
nWeights = 7
# number of trials used for the experiment
trials = 1 



# reset the random generator
#Random.seed!(12)

println("  number of variables  : ", n)
println("  number of objectives : ", o)
println("  reference point      : ", rp)
println("  number of weights    : ", listnbrWeights)
println("  number of instances  : ", nInstances)
println("  solver MIP invoked   : ", solver)

allareH̃ = (Float64)[]    # all average_relative_error on estimation of H 
allaCPUt = (Float64)[]   # all average_value of CPUt 

All_average_relative_error_Hestimated = (Float64)[]

W = 0.0                       # average precision

oneExpe7Weights = resultsExpe(  listnbrWeights,
                        0.0, 
                        zeros(7),
                        zeros(7),
                        zeros(7),
                        zeros(7),
                        zeros(7),
                        zeros(7)                        
                    )


resS = zeros(Int64, nInstances)
resCPUtS = zeros(Float64, nInstances)
resH = zeros(Float64, nInstances)


resH̃ = zeros(Float64, nInstances, nWeights)
res_H_estimated_normalized = zeros(Float64, nInstances, nWeights)
res_relative_error_Hestimated = zeros(Float64, nInstances, nWeights)
res_lowerCI = zeros(Float64, nInstances, nWeights)
res_upperCI = zeros(Float64, nInstances, nWeights)
resCPUtH̃ = zeros(Float64, nInstances, nWeights)

average_relative_error_Hestimated = zeros(Float64, nWeights)

#res_relative_error_Hestimated = zeros(Float64, nWeights)

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

    # lists saving results collected over resolutions
    #listH = (Float64)[]
    #listH̃ = (Float64)[]
    #list_H_estimated_normalized = (Float64)[]
    #list_lowerCI = (Float64)[]
    #list_upperCI = (Float64)[]
    #listCPUt = (Float64)[]
    #list_relative_error_Hestimated = (Float64)[]


    for iInstance in 1:nInstances

        # ==== PART 0: INSTANCE ===================================================

        println("\n---- instance $iInstance -------------------------------")

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

        resS[iInstance] = cardS
        resCPUtS[iInstance] = t_elapsedS

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
        oneExpe7Weights.Hmeasure = Hmeasure

        write(ioAll, string("H(S) = ",Hmeasure, " \n"))
        write(ioAll, string("\n"))

        resH[iInstance] = Hmeasure

        # ==== Part 2: ESTIMATION =====================================================

        # =============================================================================
        println("\nCompute the estimation of H with  rp=$rp and  $nWeights weight :")

        for iWeight in 1:length(listnbrWeights)
            nbrWeights = listnbrWeights[iWeight]


            # lists saving results collected over trials
        #    listH̃ = (Float64)[]
        #    list_H_estimated_normalized = (Float64)[]
        #    list_lowerCI = (Float64)[]
        #    list_upperCI = (Float64)[]
        #    listCPUt = (Float64)[]
        #    list_relative_error_Hestimated = (Float64)[]


            startH = time()
            H̃, info_for_CI = Hrevised3(Hmeasure, solver, p,w,c, rp, nbrWeights)                
            t_elapsedH = round(time() - startH, digits=2)

            # compute relative error on H_estimated_normalized
            #push!(list_relative_error_Hestimated, abs( 1.0 - info_for_CI[1]) )                    

            @printf("  weights = %5d ", nbrWeights)
            @printf(" H_est = %.1f ", round(H̃, digits=2) )
            @printf(" H_est_normalized = %.5f ", round(info_for_CI[1], digits=5) )
            @printf(" CI for 95%% = [%.5f, %.5f] ", round(info_for_CI[2], digits=5), round(info_for_CI[3], digits=5) )
            @printf(" Relative error H_est = %.5f ", abs( 1.0 - info_for_CI[1]))
            println(" t_elapsed = $t_elapsedH s")

            # save results obtained with 1 trial
            #push!(listH̃, H̃)
            #push!(list_H_estimated_normalized, info_for_CI[1])
            #push!(list_lowerCI, info_for_CI[2])
            #push!(list_upperCI, info_for_CI[3])
            #push!(listCPUt, t_elapsedH)

#            write(ioAll, string(nbrWeights, " ", round(H̃, digits=2), " ",t_elapsedH, "s\n"))


            resH̃[iInstance,iWeight]                          =  H̃
            res_H_estimated_normalized[iInstance,iWeight]    =  info_for_CI[1]
            res_relative_error_Hestimated[iInstance,iWeight] =  abs( 1.0 - info_for_CI[1])
            res_lowerCI[iInstance,iWeight]                   =  info_for_CI[2]
            res_upperCI[iInstance,iWeight]                   =  info_for_CI[3]
            resCPUtH̃[iInstance,iWeight]                      =  t_elapsedH

        end # several weights
    end # several instances

    #@show resS
    #@show resCPUtS
    #@show resH

    #@show resH̃
    #@show res_H_estimated_normalized
    #@show res_lowerCI
    #@show res_upperCI
    #@show res_relative_error_Hestimated
    #@show resCPUtH̃


            write(ioAll, string("\n"))

    # compute average value of list_relative_error_Hestimated
    average_H_estimated_normalized    = [average_value(res_H_estimated_normalized[:,i]) for i=1:nWeights]
    average_CPUtH̃                     = [average_value(resCPUtH̃[:,i]) for i=1:nWeights]
    average_lowerCI                   = [average_value(res_lowerCI[:,i]) for i=1:nWeights]
    average_upperCI                   = [average_value(res_upperCI[:,i]) for i=1:nWeights]
    average_relative_error_Hestimated = [average_value(res_relative_error_Hestimated[:,i]) for i=1:nWeights] 

    @show average_relative_error_Hestimated



    println("\nSummary of results...")
    println("  S                                       =  ", resS)            
    println("  CPUt for computing S                    =  ", resCPUtS)
    #println("  value H(S)                              =  ", resH)

    #println("  value H̃(S)                              =  ", resH̃)
    #println("  average value of H̃                      =  ", average_CPUtH̃)   
    println("  average CPUt for H̃                      =  ", average_CPUtH̃)    

    println("  average value of upper CI               =  ", average_upperCI)
    println("  average value of H_estimated_normalized =  ", average_H_estimated_normalized)        
    println("  average value of lower CI               =  ", average_lowerCI) 

    println("  Average relative error Hestimated       =  ", average_relative_error_Hestimated)

    #@assert false "stop"

    #write(ioAll, string("average value H̃             = ",avH̃, " \n"))
    #write(ioAll, string("average CPUt for H̃          = ",avCPUt, " \n\n"))

    #        push!(All_average_relative_error_Hestimated, average_relative_error_Hestimated)            

            # -------------------------------------------------------------------------

         #   avH̃ = average_value(listH̃)
         #   avCPUt = average_value(listCPUt)
         #   push!(allaCPUt, avCPUt)

            #aaeH̃ = average_absolue_error(Hmeasure, listH̃)
            #areH̃ = average_relative_error(Hmeasure, listH̃)
         

            #=
            println("\nSummary of results...")
            @printf("  S                                       = %1.6e \n", Hmeasure)            
            @printf("  value H(S)                              = %1.6e \n", Hmeasure)
            @printf("  average value H̃                         = %1.6e \n", avH̃)
            @printf("  CPUt for computing S                    = %.2f s\n", t_elapsedS)

            @printf("  average CPUt for H̃                      = %.2f s\n", avCPUt)    

            @printf("  average value of upper CI               = %1.6e \n", oneExpe7Weights.CIhightLw[iWeight])
            @printf("  average value of H_estimated_normalized = %1.6e \n", oneExpe7Weights.avLw[iWeight])        
            @printf("  average value of lower CI               = %1.6e \n", oneExpe7Weights.CIlowLw[iWeight]) 

            println("  Average relative error Hestimated       = ", average_relative_error_Hestimated)
            #println("  All average relative error Hestimated   = ", All_average_relative_error_Hestimated)

            write(ioAll, string("average value H̃             = ",avH̃, " \n"))
            write(ioAll, string("average CPUt for H̃          = ",avCPUt, " \n\n"))
            =#

            # Writing results on file...
            print(fresults," n  &  o  &  avg_CPUt  &  weight  &  avg_H_estimated_normalized  &  avg_CPUtH̃  &  avg_lowerCI  &  avg_upperCI  &  avg_relative_error_Hestimated \n")
            for iWeight in 1:nWeights
                print(fresults," $n  &  $o  & ")
                @printf(fresults," %.2f  & ", average_value(resCPUtS))
                print(fresults," $(listnbrWeights[iWeight])  & ")
            #    @printf(fresults," %1.6e  & ", resH[iInstance])
               # @printf(fresults," %1.6e  & ", avH̃) 
                @printf(fresults," %.6f  & ",  average_H_estimated_normalized[iWeight])
                @printf(fresults," %.2f  & ",  average_CPUtH̃[iWeight])
                @printf(fresults," %1.6e  & ", average_lowerCI[iWeight])
                @printf(fresults," %1.6e  & ", average_upperCI[iWeight])
                @printf(fresults," %.6f \n",   average_relative_error_Hestimated[iWeight])      
            end      


            # Writing results on file...
        #    print(fresults," $n  &  $o  &  $cardS  & ")
        #    @printf(fresults," %.2f  & ", t_elapsedS)
        #    @printf(fresults," %1.6e  & ", Hmeasure)
        #    print(fresults," $nbrWeights  & ")
        #    @printf(fresults," %1.6e  & ", avH̃) 
        #    @printf(fresults," %.2f  & ",  avCPUt)
        #    @printf(fresults," %1.6e  & ", oneExpe7Weights.CIlowLw[iWeight])
        #    @printf(fresults," %1.6e  & ", oneExpe7Weights.CIhightLw[iWeight])
        #    @printf(fresults," %.6f \n",   average_relative_error_Hestimated)             
            #println(f, "Contenu")



        #    oneExpe7Weights.avH̃[iWeight]       = 0.0 #avH̃
        #    oneExpe7Weights.CIhight[iWeight]   = 0.0 
        #    oneExpe7Weights.CIlow[iWeight]     = 0.0 

            oneExpe7Weights.avLw      =  deepcopy(average_H_estimated_normalized)  # average value of H_estimated_normalized
            oneExpe7Weights.CIlowLw   =  deepcopy(average_lowerCI)                 # average value of lower CI
            oneExpe7Weights.CIhightLw =  deepcopy(average_upperCI)                 # average value of upper CI   

    #    end # several weights

#    end # several instances

# ==== PLOTS ==================================================================


    plot(listnbrWeights, average_relative_error_Hestimated, #All_average_relative_error_Hestimated, 
        seriestype = :line, 
        marker = :circle,
        title = string(n)*" variables | "*string(o)*" objectives", # | "*string(trials)*" trial(s)",
        xlabel = "Number of weight vectors/iterations",
        ylabel = "avg. rel. err. on H estimated normalized",
        legend = false,
        linewidth = 2,
        xticks = listnbrWeights,
        xrotation = 45,
        show = true
    )

    savefig("kp-"*string(n)*"-"*string(o))

    #plot_valuesCI2(oneExpe7Weights)
    plot_valuesCI3(oneExpe7Weights, res_H_estimated_normalized)
    savefig("H"*string(n)*"-"*string(o))

end

# closing the file collecting all results
close(fresults)


#println("\n\nAll average relative error Hestimated = ", All_average_relative_error_Hestimated)
#println("All CPUt with ", solver, " = ", allaCPUt)


#println("\nAll average value of upper CI               = ", oneExpe7Weights.CIhightLw)
#println("All average value of H_estimated_normalized = ", oneExpe7Weights.avLw)
#println("All average value of upper CI               = ", oneExpe7Weights.CIlowLw)   




nothing

#=
function plot_valuesCI2(oneExpe7Weights::resultsExpe)

#    exact = fill(oneExpe7Weights.Hmeasure,7)
    exact = fill(1,7)   

    yerr_low = oneExpe7Weights.avLw .- oneExpe7Weights.CIlowLw
    yerr_high = oneExpe7Weights.CIhightLw .- oneExpe7Weights.avLw 

    plot(listnbrWeights, oneExpe7Weights.avLw, yerror = (yerr_low, yerr_high),
        label = "Average ± CI", lw=2, marker=:circle, color=:red,
        xticks = (listnbrWeights, string.(listnbrWeights)), xrotation = 45
    )

    plot!(listnbrWeights, exact, label = "Exact", marker=:diamond, ms=6, color=:black, linestyle=:dash)

    xlabel!("Number of weight vectors/iterations")
    ylabel!("H estimated normalized")
    title!(string(n)*" variables | "*string(o)*" objectives | "*string(trials)*" trial(s) | CI 95%")
end

plot_valuesCI2(oneExpe7Weights)
savefig("H"*string(n)*"-"*string(o))
=#
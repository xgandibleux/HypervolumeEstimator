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
n = 20  
# number of objectives
o = 2      

# number of instances in the experiment 1 or 7
nInstances = 10
# reference point used for the H measure
rp = zeros(Int,o)
# list of number of weights to use
listnbrWeights = fill(1000,nInstances)
# number of trials used for the experiment
trials = 1 


# reset the random generator
Random.seed!(1234)

println("  number of variables  : ", n)
println("  number of objectives : ", o)
println("  reference point      : ", rp)
println("  number of weights    : ", listnbrWeights[1])
println("  number of instances  : ",nInstances)
println("  solver MIP invoked   : ", solver)
println("  number of trials     : ", trials)

allareH̃ = (Float64)[]    # all average_relative_error on estimation of H 
allaCPUt = (Float64)[]   # all average_value of CPUt 

All_average_relative_error_Hestimated = (Float64)[]

W = 0.0                       # average precision

oneExpe = resultsExpe(  collect(1:nInstances), 
                        0.0, 
                        zeros(nInstances),
                        zeros(nInstances),
                        zeros(nInstances),
                        zeros(nInstances),
                        zeros(nInstances),
                        zeros(nInstances)                        
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
        for iWeight in 1:1#length(listnbrWeights)
            nbrWeights = listnbrWeights[instanceID]

            # -------------------------------------------------------------------------
            println("\nCompute the estimation of H with  rp=$rp and  $nbrWeights weight :\n")

            # lists saving results collected over trials
            listH̃ = (Float64)[]
            list_H_estimated_normalized = (Float64)[]
            list_lowerCI = (Float64)[]
            list_upperCI = (Float64)[]
            listCPUt = (Float64)[]
            list_relative_error_Hestimated = (Float64)[]

            for _ in 1:trials
                startH = time()
                H̃, info_for_CI = Hrevised3(Hmeasure, solver, p,w,c, rp, nbrWeights)                
                t_elapsedH = round(time() - startH, digits=2)

                # compute relative error on H_estimated_normalized
                push!(list_relative_error_Hestimated, abs( 1.0 - info_for_CI[1]) )                    

                @printf("  H_estimated = %.1f ", round(H̃, digits=2) )
                @printf(" H_estimated_normalized = %.5f ", round(info_for_CI[1], digits=5) )
                @printf(" Confidence interval for 95%% = [%.5f, %.5f] ", round(info_for_CI[2], digits=5), round(info_for_CI[3], digits=5) )
                println(" t_elapsed = $t_elapsedH s")

                # save results obtained with 1 trial
                push!(listH̃, H̃)
                push!(list_H_estimated_normalized, info_for_CI[1])
                push!(list_lowerCI, info_for_CI[2])
                push!(list_upperCI, info_for_CI[3])
                push!(listCPUt, t_elapsedH)

                write(ioAll, string(nbrWeights, " ", round(H̃, digits=2), " ",t_elapsedH, "s\n"))

            end # several trials

            write(ioAll, string("\n"))

            # compute average value of list_relative_error_Hestimated
            average_relative_error_Hestimated = average_value(list_relative_error_Hestimated)
            push!(All_average_relative_error_Hestimated, average_relative_error_Hestimated)            

            # -------------------------------------------------------------------------

            avH̃ = average_value(listH̃)
            avCPUt = average_value(listCPUt)
            push!(allaCPUt, avCPUt)

            #aaeH̃ = average_absolue_error(Hmeasure, listH̃)
            #areH̃ = average_relative_error(Hmeasure, listH̃)

            oneExpe.avH̃[instanceID]       = 0.0 #avH̃
            oneExpe.CIhight[instanceID]   = 0.0 
            oneExpe.CIlow[instanceID]     = 0.0 

            oneExpe.avLw[instanceID]      = average_value(list_H_estimated_normalized)  # average value of H_estimated_normalized
            oneExpe.CIlowLw[instanceID]   = average_value(list_lowerCI)                 # average value of lower CI
            oneExpe.CIhightLw[instanceID] = average_value(list_upperCI)                 # average value of upper CI            

            println("\nSummary of results...")
            @printf("  value H(S)                              = %1.6e \n", Hmeasure)
            @printf("  average value H̃                         = %1.6e \n", avH̃)
            @printf("  CPUt for computing S                    = %.2f s\n", t_elapsedS)
            @printf("  average CPUt for H̃                      = %.2f s\n", avCPUt)    
            @printf("  average value of upper CI               = %1.6e \n", oneExpe.CIhightLw[instanceID])
            @printf("  average value of H_estimated_normalized = %1.6e \n", oneExpe.avLw[instanceID])        
            @printf("  average value of lower CI               = %1.6e \n", oneExpe.CIlowLw[instanceID]) 
            println("  Average relative error Hestimated       = ", average_relative_error_Hestimated)
            #println("  All average relative error Hestimated   = ", All_average_relative_error_Hestimated)

            write(ioAll, string("average value H̃             = ",avH̃, " \n"))
            write(ioAll, string("average CPUt for H̃          = ",avCPUt, " \n\n"))

            # Writing results on file...
            print(fresults," $n  &  $o  &  $cardS  & ")
            @printf(fresults," %.2f  & ", t_elapsedS)
            @printf(fresults," %1.6e  & ", Hmeasure)
            print(fresults," $nbrWeights  & ")
            @printf(fresults," %1.6e  & ", avH̃) 
            @printf(fresults," %.2f  & ",  avCPUt)
            @printf(fresults," %1.6e  & ", oneExpe.CIlowLw[instanceID])
            @printf(fresults," %1.6e  & ", oneExpe.CIhightLw[instanceID])
            @printf(fresults," %.6f \n",   average_relative_error_Hestimated)
            #println(f, "Contenu")

        end # several weights

    end # several instances

end

# closing the file collecting all results
close(fresults)


println("\n\nAll average relative error Hestimated = ", All_average_relative_error_Hestimated)
println("All CPUt with ", solver, " = ", allaCPUt)

println("\nAll average value of upper CI               = ", oneExpe.CIhightLw)
println("All average value of H_estimated_normalized = ", oneExpe.avLw)
println("All average value of upper CI               = ", oneExpe.CIlowLw)


# ==== PLOTS ==================================================================


plot(collect(1:nInstances), All_average_relative_error_Hestimated, 
    seriestype = :line, 
    marker = :circle,
    title = string(n)*" variables | "*string(o)*" objectives | "*string(trials)*" trials",
    xlabel = "Instances",
    ylabel = "avg. rel. err. on H estimated normalized",
    legend = false,
    linewidth = 2,
    xticks = (collect(1:nInstances), string.(collect(1:nInstances))),
    xrotation = 45,
    show = true
)

savefig("kp-"*string(n)*"-"*string(o))
nothing


function plot_valuesCI2(oneExpe::resultsExpe)

#    exact = fill(oneExpe.Hmeasure,7)
    exact = fill(1,nInstances)   

    yerr_low = oneExpe.avLw .- oneExpe.CIlowLw
    yerr_high = oneExpe.CIhightLw .- oneExpe.avLw 

    plot(collect(1:nInstances), oneExpe.avLw, yerror = (yerr_low, yerr_high),
         label = "Average ± CI", lw=2, marker=:circle, color=:red,
         xticks = (collect(1:nInstances), string.(collect(1:nInstances))), xrotation = 45
    )

    plot!(collect(1:nInstances), exact, label = "Exact", marker=:diamond, ms=6, color=:black, linestyle=:dash)

    xlabel!("Instances")
    ylabel!("H estimated normalized")
    title!(string(n)*" variables | "*string(o)*" objectives | "*string(trials)*" trials | CI 95%")
end


plot_valuesCI2(oneExpe)
savefig("H"*string(n)*"-"*string(o))


# -------------------------

nVar = n

using Statistics

# --- Data ---
erreurs = copy(All_average_relative_error_Hestimated)


n    = length(erreurs)
μ    = mean(erreurs)
σ    = std(erreurs; corrected=false)
idx  = 1:n

# --- Colour palette ---
bleu   = RGB(0.208, 0.541, 0.871)   # #378ADD
orange = RGB(0.729, 0.459, 0.090)   # #BA7517
vert   = RGB(0.114, 0.620, 0.459)   # #1D9E75
rouge  = RGB(0.847, 0.353, 0.188)   # #D85A30
gris   = RGB(0.47, 0.47, 0.44)

# ── Summary statistics ────────────────────────────────────────────────────────
println("─────────────────────────────────────")
println("  Statistics — Relative error H")
println("─────────────────────────────────────")
@printf("  Mean        : %.6f\n", μ)
@printf("  Std dev     : %.6f\n", σ)
@printf("  Minimum     : %.6f  (measurement %d)\n", minimum(erreurs), argmin(erreurs))
@printf("  Maximum     : %.6f  (measurement %d)\n", maximum(erreurs), argmax(erreurs))
println("─────────────────────────────────────")

# ── Main figure: 2 subplots ───────────────────────────────────────────────────
theme(:default)
gr()

# ── Subplot 1: Time series ────────────────────────────────────────────────────
p1 = plot(
    idx, erreurs;
    seriestype  = :line,
    linecolor   = bleu,
    linewidth   = 2,
    marker      = :circle,
    markersize  = 5,
    markercolor = bleu,
    markerstrokecolor = :white,
    markerstrokewidth = 1.2,
    label       = "Relative error",
    xlabel      = "Measurement index",
    ylabel      = "Average relative error",
    title       = "Relative error over measurements — Estimated H",
    titlefontsize = 11,
    guidefontsize = 9,
    tickfontsize  = 8,
    legendfontsize = 8,
    legend      = :topright,
    grid        = true,
    gridalpha   = 0.25,
    framestyle  = :box,
    xticks      = 1:n,
    ylims       = (max(0, minimum(erreurs) - 3σ), maximum(erreurs) + 3σ),
)

# ± 1σ band
plot!(p1,
    idx, fill(μ + σ, n);
    fillrange   = fill(μ - σ, n),
    fillalpha   = 0.12,
    fillcolor   = bleu,
    linewidth   = 0,
    label       = "± 1 std dev",
)

# ± 1σ dashed boundaries
plot!(p1, idx, fill(μ + σ, n);
    linecolor = bleu, linewidth = 1, linestyle = :dash, linealpha = 0.5, label = "")
plot!(p1, idx, fill(μ - σ, n);
    linecolor = bleu, linewidth = 1, linestyle = :dash, linealpha = 0.5, label = "")

# Mean line
hline!(p1, [μ];
    linecolor = orange, linewidth = 1.8, linestyle = :dash,
    label     = "Mean ($(round(μ; digits=5)))")

# Annotate extremes
annotate!(p1,
    argmin(erreurs), minimum(erreurs) - 0.4σ,
    text("min", 7, vert, :center))
annotate!(p1,
    argmax(erreurs), maximum(erreurs) + 0.4σ,
    text("max", 7, rouge, :center))

# ── Subplot 2: Histogram ──────────────────────────────────────────────────────
nbins = round(Int, 1 + log2(n))   # Sturges rule
p2 = histogram(
    erreurs;
    bins        = nbins,
    color       = bleu,
    alpha       = 0.6,
    linecolor   = bleu,
    linewidth   = 0.8,
    label       = "Measurements",
    xlabel      = "Average relative error",
    ylabel      = "Count",
    title       = "Error distribution",
    titlefontsize = 11,
    guidefontsize = 9,
    tickfontsize  = 8,
    legendfontsize = 8,
    legend      = :topright,
    grid        = true,
    gridalpha   = 0.25,
    framestyle  = :box,
)

# Mean line
vline!(p2, [μ];
    linecolor = orange, linewidth = 2, linestyle = :dash,
    label     = "Mean")

# ── Final layout ──────────────────────────────────────────────────────────────
fig = plot(p1, p2;
    layout  = (2, 1),
    size    = (800, 620),
    margin  = 6Plots.mm,
    bottom_margin = 4Plots.mm,
)



savefig(fig, "several"*string(nVar)*"-"*string(o)*".png")
display(fig)

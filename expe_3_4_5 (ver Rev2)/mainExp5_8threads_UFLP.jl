#= EXPERIMENT 1 — UFLP:

  - given nI (users), nJ (services) and o (objectives)
    - warm-up: JIT compilation on a minimal instance (not timed)
    - generate nInstances instances randomly
    - for each instance:
        - compute rp : reference point via UB bound (negated for estimator)
        - compute Y_N : exact nondominated set (Tamby-Vanderpooten via MOA)
        - measure H exactly via @ccall to fpli_hv (libfpli_hv.dylib)
        - for each value in listnbrWeights:
            - estimate H with Hrevised3_UFLP (L_threaded_UFLP, Threads.@threads)
            - collect elapsed time, normalized value, CI 95%
    - compute across instances:
        - average H_estimated_normalized
        - average confidence interval 95%
        - average relative error
        - average CPU time
  - report:
    - cardinality of Y_N, elapsed time for computing Y_N
    - H exact, reference point rp (in minimisation space)
    - H estimated (normalized, CI, relative error, CPU time)
=#

using Printf
using Random
using Base.Threads
using JuMP, Gurobi
import MultiObjectiveAlgorithms as MOA
using Distributions
using SpecialFunctions
using HypothesisTests
using Statistics
using Plots

Random.seed!(12345)

include("src/common.jl")
include("src/instanceUFLP.jl")
include("src/solveUFLP.jl")
include("src/files.jl")
include("src/estimHyperVol_UFLP.jl")
include("src/analyze.jl")

println("-"^80)

mutable struct resultsExpe
    x          :: Vector{Int64}
    Hmeasure   :: Float64
    avLw       :: Vector{Float64}
    CIhightLw  :: Vector{Float64}
    CIlowLw    :: Vector{Float64}
end


function plot_valuesCI(oneExpe7Weights::resultsExpe, res_H_estimated_normalized, nI, nJ, o)
    exact = fill(1, 7)
    yerr_low  = oneExpe7Weights.avLw .- oneExpe7Weights.CIlowLw
    yerr_high = oneExpe7Weights.CIhightLw .- oneExpe7Weights.avLw

    plot(listnbrWeights, exact,
        label = "Exact H", marker=:diamond, ms=6, color=:black, linestyle=:dash)

    n_rows = size(res_H_estimated_normalized, 1)
    x_scattered = repeat(listnbrWeights, inner = n_rows)
    y_scattered = vec(res_H_estimated_normalized)

    scatter!(x_scattered, y_scattered,
        label = "Individual H estimated",
        color=:blue, alpha=0.75, marker=:hline, ms=6, markerstrokewidth=1)

    plot!(listnbrWeights, oneExpe7Weights.avLw, yerror = (yerr_low, yerr_high),
        label = "Average H estimated ± average CI",
        lw=2, marker=:circle, color=:red,
        xticks = (listnbrWeights, string.(listnbrWeights)), xrotation = 45)

    xlabel!("Number of weight vectors/iterations")
    ylabel!("H estimated normalized")
    title!("UFLP $nI×$nJ | $o objectives | CI 95%")
end


# =============================================================================
println("Setup the parameters...")

solver = Gurobi.Optimizer   # Gurobi : best performance on larger instances with @threads

# number of users and services
nI = 25  # clients
nJ = 10  # services
# number of objectives
o  = 3

# number of instances
nInstances = 10
# list of number of weights to use
listnbrWeights = [100, 500, 1000, 1500, 2000, 5000, 10000]
nWeights = 7

println("  number of users      : ", nI)
println("  number of services   : ", nJ)
println("  number of objectives : ", o, " (minimisation, negated for estimator)")
println("  reference point      : computed per instance via UB bound")
println("  number of weights    : ", listnbrWeights)
println("  number of instances  : ", nInstances)
println("  solver MIP invoked   : ", solver)
println("  threads available    : ", Threads.nthreads())

oneExpe7Weights = resultsExpe(
    listnbrWeights,
    0.0,
    zeros(7),
    zeros(7),
    zeros(7)
)

resS      = zeros(Int64,   nInstances)
resCPUtS  = zeros(Float64, nInstances)
resH      = zeros(Float64, nInstances)

resH̃                          = zeros(Float64, nInstances, nWeights)
res_H_estimated_normalized    = zeros(Float64, nInstances, nWeights)
res_relative_error_Hestimated = zeros(Float64, nInstances, nWeights)
res_lowerCI                   = zeros(Float64, nInstances, nWeights)
res_upperCI                   = zeros(Float64, nInstances, nWeights)
resCPUtH̃                      = zeros(Float64, nInstances, nWeights)

average_relative_error_Hestimated = zeros(Float64, nWeights)

#
# File collecting all results for a table in the paper
#
fresults     = open("uflp-tableResultsExpe1.res", "a")
instanceName = "uflp-" * string(nI) * "x" * string(nJ) * "-" * string(o)

# =============================================================================
# Warm-up: force JIT compilation on a minimal instance before the experiment.
# The first call to each function triggers compilation; timings are not recorded.
println("Warm-up (JIT compilation)...")
let inst_w = generate_UFLP(2, 2, o)
    rp_w    = reference_point(inst_w)
    S_w, _  = solve_UFLP(solver, inst_w)
    if length(S_w) > 0
        rp_hv_w = Float64.(-rp_w)
        compute_Hmeasure_min(S_w, inst_w.o, rp_hv_w)
        Hrevised3_UFLP(1.0, solver, inst_w, rp_w, 10)
    end
end
println("Warm-up done.")
println("-"^80)

open(instanceName * ".res", "w") do ioAll
    write(ioAll, string(instanceName, "\n"))

    for iInstance in 1:nInstances

        # ==== PART 0: INSTANCE ===================================================
        println("\n---- instance $iInstance -------------------------------")

        println("\nGenerate a multi-objective UFLP instance...")
        inst = generate_UFLP(nI, nJ, o)

        # reference point computed per instance (UB bound, negated for estimator)
        rp = reference_point(inst)
        println("  reference point rp = ", -rp, " (in minimisation space)")
        write(ioAll, string("rp = ", -rp, " (in minimisation space)\n"))

        # ==== PART 1: EXACT ======================================================
        println("\nCompute S = Y_N, the set of nondominated points...")
        start = time()
        S, cardS = solve_UFLP(solver, inst)
        save_nondominatedpoints(instanceName * ".yn", S)
        t_elapsedS = round(time() - start, digits=2)
        println("  |S|  = ", cardS, " ($t_elapsedS s)")
        write(ioAll, string("|S|  = ", cardS, " ($t_elapsedS s) \n"))

        resS[iInstance]     = cardS
        resCPUtS[iInstance] = t_elapsedS

        println("\nCompute H, the hypervolume measure...")
        # rp_hv = [-rp[k]] in minimisation space matches rp = [rp[k]] used by estimator
        rp_hv = Float64.(-rp)
        Hmeasure = compute_Hmeasure_min(S, inst.o, rp_hv)
        @printf("  H(S) = %1.6e\n", Hmeasure)
        oneExpe7Weights.Hmeasure = Hmeasure
        write(ioAll, string("H(S) = ", Hmeasure, " \n\n"))
        resH[iInstance] = Hmeasure

        # ==== PART 2: ESTIMATION =================================================
        println("\nCompute the estimation of H with rp=$(-rp) and $nWeights weight sets:")

        for iWeight in 1:length(listnbrWeights)
            nbrWeights = listnbrWeights[iWeight]

            startH = time()
            H̃, info_for_CI = Hrevised3_UFLP(Hmeasure, solver, inst, rp, nbrWeights)
            t_elapsedH = round(time() - startH, digits=2)

            @printf("  weights = %5d ", nbrWeights)
            @printf(" H_est = %.1f ", round(H̃, digits=2))
            @printf(" H_est_normalized = %.5f ", round(info_for_CI[1], digits=5))
            @printf(" CI for 95%% = [%.5f, %.5f] ",
                round(info_for_CI[2], digits=5), round(info_for_CI[3], digits=5))
            @printf(" Relative error H_est = %.5f ", abs(1.0 - info_for_CI[1]))
            println(" t_elapsed = $t_elapsedH s")

            resH̃[iInstance, iWeight]                          = H̃
            res_H_estimated_normalized[iInstance, iWeight]    = info_for_CI[1]
            res_relative_error_Hestimated[iInstance, iWeight] = abs(1.0 - info_for_CI[1])
            res_lowerCI[iInstance, iWeight]                   = info_for_CI[2]
            res_upperCI[iInstance, iWeight]                   = info_for_CI[3]
            resCPUtH̃[iInstance, iWeight]                      = t_elapsedH

        end # weights loop
    end # instances loop

    write(ioAll, string("\n"))

    # ==== AVERAGES ACROSS INSTANCES ==============================================
    average_H_estimated_normalized    = [average_value(res_H_estimated_normalized[:, i])    for i = 1:nWeights]
    average_CPUtH̃                     = [average_value(resCPUtH̃[:, i])                      for i = 1:nWeights]
    average_lowerCI                   = [average_value(res_lowerCI[:, i])                   for i = 1:nWeights]
    average_upperCI                   = [average_value(res_upperCI[:, i])                   for i = 1:nWeights]
    average_relative_error_Hestimated = [average_value(res_relative_error_Hestimated[:, i]) for i = 1:nWeights]

    @show average_relative_error_Hestimated

    println("\nSummary of results...")
    println("  S                                       =  ", resS)
    println("  CPUt for computing S                    =  ", resCPUtS)
    println("  average CPUt for H̃                      =  ", average_CPUtH̃)
    println("  average value of upper CI               =  ", average_upperCI)
    println("  average value of H_estimated_normalized =  ", average_H_estimated_normalized)
    println("  average value of lower CI               =  ", average_lowerCI)
    println("  Average relative error Hestimated       =  ", average_relative_error_Hestimated)

    # ==== WRITE TABLE FILE =======================================================
    print(fresults, " nI  &  nJ  &  o  &  avg_CPUt_S  &  weight  &  avg_H_estimated_normalized  &  avg_CPUtH̃  &  avg_lowerCI  &  avg_upperCI  &  avg_relative_error_Hestimated \n")
    for iWeight in 1:nWeights
        print(fresults, " $nI  &  $nJ  &  $o  & ")
        @printf(fresults, " %.2f  & ", average_value(resCPUtS))
        print(fresults, " $(listnbrWeights[iWeight])  & ")
        @printf(fresults, " %.6f  & ",  average_H_estimated_normalized[iWeight])
        @printf(fresults, " %.2f  & ",  average_CPUtH̃[iWeight])
        @printf(fresults, " %1.6e  & ", average_lowerCI[iWeight])
        @printf(fresults, " %1.6e  & ", average_upperCI[iWeight])
        @printf(fresults, " %.6f \n",   average_relative_error_Hestimated[iWeight])
    end

    # ==== PLOTS ==================================================================
    oneExpe7Weights.avLw      = deepcopy(average_H_estimated_normalized)
    oneExpe7Weights.CIlowLw   = deepcopy(average_lowerCI)
    oneExpe7Weights.CIhightLw = deepcopy(average_upperCI)

    plot(listnbrWeights, average_relative_error_Hestimated,
        seriestype = :line,
        marker = :circle,
        title = "UFLP $nI×$nJ | $o objectives",
        xlabel = "Number of weight vectors/iterations",
        ylabel = "avg. rel. err. on H estimated normalized",
        legend = false,
        linewidth = 2,
        xticks = listnbrWeights,
        xrotation = 45,
        show = true
    )
    savefig("uflp-" * string(nI) * "x" * string(nJ) * "-" * string(o))

    plot_valuesCI(oneExpe7Weights, res_H_estimated_normalized, nI, nJ, o)
    savefig("uflp-H-" * string(nI) * "x" * string(nJ) * "-" * string(o))

end

close(fresults)

nothing

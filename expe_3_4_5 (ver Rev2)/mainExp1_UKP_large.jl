#= EXPERIMENT 1 — UKP (large instances, no exact hypervolume):

  - given a value for n (variables) and o (objectives)
    - warm-up: JIT compilation on a minimal instance (not timed)
    - generate ONE instance (fixed)
    - compute rp once on this instance
    - for nRuns runs (each with a different RNG seed):
        - for each value in listnbrWeights:
            - estimate H with Hrevised3_nonorm (L_threaded, Threads.@threads)
            - collect elapsed time, H estimated (raw), CI 95% (intra-run)
    - compute across runs:
        - average H_estimated
        - average CI 95% (average of intra-run CIs)
        - average CPU time
  - report:
    - H estimated (raw, CI, CPU time) per N
    - graphique: nRuns estimations per N show RNG variability on a fixed instance
=#

using Printf
using Random
using Base.Threads
using JuMP, Gurobi
using SpecialFunctions
using HypothesisTests
using Statistics
using Plots

include("src/common.jl")
include("src/instance_UKP.jl")
include("src/files.jl")
include("src/estimHyperVol_UKP.jl")
include("src/analyze.jl")
include("src/estimHyperVol_UKP_nonorm.jl")

println("-"^80)

mutable struct resultsExpe_large
    x         :: Vector{Int64}
    avH̃       :: Vector{Float64}
    CIhightLw :: Vector{Float64}
    CIlowLw   :: Vector{Float64}
end


function plot_valuesCI_large(oneExpe::resultsExpe_large, res_H̃, weights)
    nRuns = size(res_H̃, 1)

    # plot individual runs first (behind the average curve)
    p = scatter(weights, res_H̃[1, :],
        label  = "Individual H estimated",
        color  = :blue, alpha = 0.75, marker = :hline, ms = 6, markerstrokewidth = 1,
        xticks = (weights, string.(weights)), xrotation = 45)
    for iRun in 2:nRuns
        scatter!(p, weights, res_H̃[iRun, :],
            label = "", color = :blue, alpha = 0.75, marker = :hline, ms = 6, markerstrokewidth = 1)
    end

    # overlay average curve (no CI bars)
    plot!(p, weights, oneExpe.avH̃,
        label  = "Average H estimated",
        lw = 2, marker = :circle, ms = 5, color = :red)

    xlabel!("Number of weight vectors/iterations")
    ylabel!("H estimated (raw)")
    title!(string(n) * " variables | " * string(o) * " objectives")
    return p
end


# =============================================================================
println("Setup the parameters...")

solver    = Gurobi.Optimizer
solver_fn = L_threaded       # L_threaded or Lbis_threaded

# reference point mode : :zero or :LB
rp_mode = :LB

# number of variables
n = 1000
# number of objectives
o = 10

# number of runs on the same fixed instance (RNG variability)
nRuns = 5
# list of number of weights to use
listnbrWeights = [100, 500, 1000, 1500, 2000, 5000, 10000]
nWeights = 7

println("  number of variables  : ", n)
println("  number of objectives : ", o)
println("  reference point mode : ", rp_mode)
println("  number of weights    : ", listnbrWeights)
println("  number of runs       : ", nRuns, " (same instance, varying RNG)")
println("  solver MIP invoked   : ", solver)
println("  resolution strategy  : ", solver_fn)

oneExpe7Weights = resultsExpe_large(
    listnbrWeights,
    zeros(nWeights),
    zeros(nWeights),
    zeros(nWeights)
)

resH̃     = zeros(Float64, nRuns, nWeights)
res_CIlo = zeros(Float64, nRuns, nWeights)
res_CIhi = zeros(Float64, nRuns, nWeights)
resCPUtH̃ = zeros(Float64, nRuns, nWeights)

#
# File collecting all results for a table in the paper
#
fresults     = open("ukp-large-tableResultsExpe1.res", "a")
instanceName = "ukp-large-" * string(n) * "-" * string(o)

# =============================================================================
# Generate ONE fixed instance
Random.seed!(1234)
inst = generate_MO01UKP(n, o)
rp   = (rp_mode == :LB) ? reference_point_LB(inst) : zeros(Int, o)
println("\nFixed instance generated.")
println("  reference point rp = ", rp)

# =============================================================================
# Warm-up: force JIT compilation on a minimal instance before the experiment.
println("Warm-up (JIT compilation)...")
let inst_w = generate_MO01UKP(2, 2)
    rp_w   = reference_point_LB(inst_w)
    Hrevised3_nonorm(solver, inst_w, rp_w, 10, 1)
end
println("Warm-up done.")
println("-"^80)

open(instanceName * ".res", "w") do ioAll
    write(ioAll, string(instanceName, "\n"))
    write(ioAll, string("rp = ", rp, "\n\n"))

    for iRun in 1:nRuns

        println("\n---- run $iRun (seed = $iRun) -------------------------------")

        # ==== ESTIMATION =========================================================
        println("Compute the estimation of H with $nWeights weight sets:")

        for iWeight in 1:length(listnbrWeights)
            nbrWeights = listnbrWeights[iWeight]

            startH = time()
            H̃, (mean_lw, CIlo, CIhi) = Hrevised3_nonorm(solver, inst, rp,
                                                           nbrWeights, iRun)
            t_elapsedH = round(time() - startH, digits=2)

            @printf("  weights = %5d  H_est = %1.6e  CI 95%% = [%1.6e, %1.6e]  t = %.2f s\n",
                nbrWeights, H̃, CIlo, CIhi, t_elapsedH)

            resH̃[iRun, iWeight]     = H̃
            res_CIlo[iRun, iWeight]  = CIlo
            res_CIhi[iRun, iWeight]  = CIhi
            resCPUtH̃[iRun, iWeight]  = t_elapsedH

        end # weights loop
    end # runs loop

    write(ioAll, string("\n"))

    # ==== AVERAGES ACROSS RUNS ===================================================
    average_H̃       = [average_value(resH̃[:, i])     for i = 1:nWeights]
    average_CPUtH̃   = [average_value(resCPUtH̃[:, i]) for i = 1:nWeights]
    # average of intra-run CIs — same principle as Hrevised3
    average_lowerCI = [average_value(res_CIlo[:, i])  for i = 1:nWeights]
    average_upperCI = [average_value(res_CIhi[:, i])  for i = 1:nWeights]

    println("\nSummary of results...")
    println("  average H̃           =  ", average_H̃)
    println("  average CPUt for H̃  =  ", average_CPUtH̃)
    println("  average lower CI    =  ", average_lowerCI)
    println("  average upper CI    =  ", average_upperCI)

    # ==== WRITE TABLE FILE =======================================================
    print(fresults, " n  &  o  &  rp_mode  &  weight  &  avg_H_est  &  avg_CPUtH̃  &  avg_lowerCI  &  avg_upperCI \n")
    for iWeight in 1:nWeights
        print(fresults, " $n  &  $o  &  $rp_mode  &  $(listnbrWeights[iWeight])  & ")
        @printf(fresults, " %1.6e  & ", average_H̃[iWeight])
        @printf(fresults, " %.2f  & ",  average_CPUtH̃[iWeight])
        @printf(fresults, " %1.6e  & ", average_lowerCI[iWeight])
        @printf(fresults, " %1.6e \\\\\n", average_upperCI[iWeight])
    end

    # ==== PLOTS ==================================================================
    oneExpe7Weights.avH̃       = deepcopy(average_H̃)
    oneExpe7Weights.CIlowLw   = deepcopy(average_lowerCI)
    oneExpe7Weights.CIhightLw = deepcopy(average_upperCI)

    plot(listnbrWeights, average_H̃,
        seriestype = :line,
        marker = :circle,
        title = string(n) * " variables | " * string(o) * " objectives",
        xlabel = "Number of weight vectors/iterations",
        ylabel = "avg. H estimated (raw)",
        legend = false,
        linewidth = 2,
        xticks = listnbrWeights,
        xrotation = 45,
        show = true
    )
    savefig("ukp-large-" * string(n) * "-" * string(o))

    plot_valuesCI_large(oneExpe7Weights, resH̃, listnbrWeights)
    savefig("ukp-large-H-" * string(n) * "-" * string(o))

end

close(fresults)

nothing

#= EXPERIMENT — UKP threads comparison:

  - given a value for n (variables) and o (objectives)
    - warm-up: JIT compilation on a minimal instance (not timed)
    - generate nInstances instances randomly
    - for each instance:
        - compute rp_LB = reference_point_LB()
        - compute Y_N : exact nondominated set (Tamby-Vanderpooten via MOA)
        - measure H exactly via @ccall to fpli_hv
        - for each value in listnbrWeights:
            - estimate H with Hrevised3 (L_threaded, Threads.@threads)
            - collect normalized value, CI 95%, relative error, CPU time
    - compute across instances:
        - average H_estimated_normalized, CI, relative error, CPU time
  - report:
    - number of threads used
    - comparison table: norm, CI, rel.err, CPU time vs N

  Run twice to compare:
    julia --threads 1 mainExp_threads_comparison.jl
    julia --threads 8 mainExp_threads_comparison.jl
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

Random.seed!(1234)

include("src/common.jl")
include("src/instance_UKP.jl")
include("src/solve_UKP.jl")
include("src/files.jl")
include("src/estimHyperVol_UKP.jl")
include("src/analyze.jl")

println("-"^80)

# =============================================================================
println("Setup the parameters...")

solver    = Gurobi.Optimizer
solver_fn = L_threaded

n = 100   # adjust for each configuration: 25 / 50 / 100
o = 2    # adjust for each configuration:  4 /  3 /   2

nInstances     = 10
listnbrWeights = [100, 500, 1000, 1500, 2000, 5000, 10000]
nWeights       = 7

nT = Threads.nthreads()   # number of threads used in this run

println("  number of variables  : ", n)
println("  number of objectives : ", o)
println("  number of instances  : ", nInstances)
println("  number of weights    : ", listnbrWeights)
println("  solver MIP invoked   : ", solver)
println("  threads              : ", nT)

# results matrices
res_norm = zeros(Float64, nInstances, nWeights)
res_err  = zeros(Float64, nInstances, nWeights)
res_CIlo = zeros(Float64, nInstances, nWeights)
res_CIhi = zeros(Float64, nInstances, nWeights)
res_time = zeros(Float64, nInstances, nWeights)

instanceName = "ukp-threads-" * string(n) * "-" * string(o) * "-T" * string(nT)
fresults     = open("ukp-tableThreadsComparison-T" * string(nT) * ".res", "a")

# =============================================================================
# Warm-up
println("Warm-up (JIT compilation)...")
let inst_w = generate_MO01UKP(2, 2)
    rp_w   = reference_point_LB(inst_w)
    S_w, _ = solve_MO01UKP(solver, inst_w)
    if length(S_w) > 0
        compute_Hmeasure(S_w, inst_w.o, rp_w)
        Hrevised3(1.0, solver, inst_w, rp_w, 10, solver_fn)
    end
end
println("Warm-up done.")
println("-"^80)

open(instanceName * ".res", "w") do ioAll
    write(ioAll, string(instanceName, " | threads = ", nT, "\n\n"))

    for iInstance in 1:nInstances

        println("\n---- instance $iInstance -------------------------------")

        # ==== INSTANCE ===========================================================
        inst = generate_MO01UKP(n, o)
        rp   = reference_point_LB(inst)
        println("  rp = ", rp)

        # ==== EXACT ==============================================================
        println("\nCompute S = Y_N...")
        S, cardS = solve_MO01UKP(solver, inst)
        println("  |S| = ", cardS)
        write(ioAll, string("|S| = ", cardS, "\n"))
        write(ioAll, string("rp  = ", rp, "\n"))

        Hmeasure = compute_Hmeasure(S, inst.o, rp)
        @printf("  H(S) = %1.6e\n", Hmeasure)
        write(ioAll, string("H(S) = ", Hmeasure, "\n\n"))

        # ==== ESTIMATION =========================================================
        println("\nEstimation with $nT thread(s):")

        for iWeight in 1:nWeights
            N = listnbrWeights[iWeight]

            startH = time()
            H̃, (H_norm, CIlo, CIhi) = Hrevised3(Hmeasure, solver, inst, rp, N, solver_fn)
            elapsed = round(time() - startH, digits=2)

            res_norm[iInstance, iWeight] = H_norm
            res_err[iInstance, iWeight]  = abs(1.0 - H_norm)
            res_CIlo[iInstance, iWeight] = CIlo
            res_CIhi[iInstance, iWeight] = CIhi
            res_time[iInstance, iWeight] = elapsed

            @printf("  N=%5d | norm=%.5f CI=[%.5f,%.5f] err=%.5f t=%.2fs\n",
                N, H_norm, CIlo, CIhi, abs(1.0 - H_norm), elapsed)
        end
    end

    write(ioAll, "\n")

    # ==== AVERAGES ===============================================================
    av_norm = [average_value(res_norm[:,i]) for i in 1:nWeights]
    av_err  = [average_value(res_err[:,i])  for i in 1:nWeights]
    av_CIw  = [average_value(res_CIhi[:,i] .- res_CIlo[:,i]) for i in 1:nWeights]
    av_time = [average_value(res_time[:,i]) for i in 1:nWeights]

    println("\n", "="^80)
    println("Summary — $n variables | $o objectives | $nT thread(s) | average over $nInstances instances")
    println("="^80)
    @printf("%-8s | %-10s %-10s %-10s %-10s\n",
            "N", "norm", "CI width", "rel.err", "time (s)")
    println("-"^80)
    for i in 1:nWeights
        @printf("%-8d | %-10.5f %-10.5f %-10.5f %-10.2f\n",
            listnbrWeights[i], av_norm[i], av_CIw[i], av_err[i], av_time[i])
    end

    # ==== WRITE TABLE ============================================================
    print(fresults, " n  &  o  &  T  &  N  &  norm  &  CI width  &  rel.err  &  time \\\\\n")
    for i in 1:nWeights
        @printf(fresults, " %d  &  %d  &  %d  &  %d  &  %.6f  &  %.6e  &  %.6f  &  %.2f \\\\\n",
            n, o, nT, listnbrWeights[i],
            av_norm[i], av_CIw[i], av_err[i], av_time[i])
    end

    # ==== PLOTS ==================================================================
    plot(listnbrWeights, av_err,
        marker=:circle, lw=2, color=:blue, legend=false,
        title="$n variables | $o objectives | $nT thread(s)",
        xlabel="Number of weight vectors/iterations",
        ylabel="avg. relative error on H estimated normalized",
        xticks=listnbrWeights, xrotation=45)
    savefig("ukp-threads-err-" * string(n) * "-" * string(o) * "-T" * string(nT))

    plot(listnbrWeights, av_time,
        marker=:circle, lw=2, color=:red, legend=false,
        title="$n variables | $o objectives | $nT thread(s)",
        xlabel="Number of weight vectors/iterations",
        ylabel="avg. CPU time (s)",
        xticks=listnbrWeights, xrotation=45)
    savefig("ukp-threads-time-" * string(n) * "-" * string(o) * "-T" * string(nT))

end

close(fresults)
nothing

#= EXPERIMENT — UKP threads comparison (-revT):

  - given a value for n (variables) and o (objectives)
    - warm-up: JIT compilation on a minimal instance (not timed)
    - generate nInstances instances randomly
    - for each instance:
        - compute rp_LB = reference_point_LB()
        - compute Y_N : exact nondominated set (Tamby-Vanderpooten via MOA)
        - measure H exactly via @ccall to fpli_hv
        - for each value in listnbrWeights:
            - estimate H with Hrevised3 (L_threaded, Threads.@threads;
              chunks of fixed size, one RNG per chunk)
            - collect normalized value, CI 95%, relative error, elapsed time
    - compute across instances:
        - average H_estimated_normalized, CI, relative error, elapsed time
  - report:
    - number of threads used
    - comparison table: norm, CI, rel.err, elapsed time vs N
  - verification (design property, Proposition of Section 4.3):
    - the normalized estimates are dumped to a per-T file; if the file of
      another run with a different number of threads is found, the two
      sets of estimates are compared and must be IDENTICAL (the sampled
      directions do not depend on the number of threads)

  Run twice to compare:
    julia --threads 1 mainExp5_threads_comparison-revT.jl
    julia --threads 8 mainExp5_threads_comparison-revT.jl
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
using DelimitedFiles

Random.seed!(1234)

include("src/common-revT.jl")
include("src/instance_UKP.jl")
include("src/solve_UKP.jl")
include("src/files.jl")
include("src/estimHyperVol_UKP-revT.jl")
include("src/analyze.jl")

println("-"^80)

# =============================================================================
println("Setup the parameters...")

solver    = Gurobi.Optimizer
solver_fn = L_threaded

n = 10   # adjust for each configuration: 25 / 50 / 100
o = 2    # adjust for each configuration:  4 /  3 /   2

nInstances     = 10
listnbrWeights = [100, 500, 1000, 1500, 2000, 5000, 10000]
nWeights       = 7

nT = Threads.nthreads()   # number of threads used in this run

println("  number of variables  : ", n)
println("  number of objectives : ", o)
println("  number of instances  : ", nInstances)
println("  number of directions : ", listnbrWeights)
println("  chunk size           : ", CHUNKSIZE)
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

# file used for the cross-thread verification of the estimates
verifName(t) = "ukp-verif-" * string(n) * "-" * string(o) * "-T" * string(t) * ".txt"

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

    # ==== CROSS-THREAD VERIFICATION ==============================================
    # Dump the full-precision normalized estimates of this run, then compare
    # with any run of the same configuration executed with a different number
    # of threads. The design property (chunk partition and chunk-seeded RNGs
    # independent of the scheduling) guarantees IDENTICAL estimates.
    writedlm(verifName(nT), res_norm)
    for t in (1, 2, 4, 8)
        t == nT && continue
        other = verifName(t)
        if isfile(other)
            res_other = readdlm(other)
            if res_norm == res_other
                println("\nVERIFICATION vs T=$t: OK — all $(nInstances)x$(nWeights) ",
                        "normalized estimates are bitwise identical.")
                write(ioAll, "\nVERIFICATION vs T=$t: identical estimates.\n")
            else
                nDiff = count(res_norm .!= res_other)
                maxD  = maximum(abs.(res_norm .- res_other))
                println("\nVERIFICATION vs T=$t: FAILED — $nDiff differing entries, ",
                        "max abs. difference = $maxD.")
                write(ioAll, "\nVERIFICATION vs T=$t: FAILED ($nDiff entries, max $maxD).\n")
            end
        end
    end

    # ==== PLOTS ==================================================================
    plot(listnbrWeights, av_err,
        marker=:circle, lw=2, color=:blue, legend=false,
        title="$n variables | $o objectives | $nT thread(s)",
        xlabel="Number of directions N",
        ylabel="avg. relative error on H estimated normalized",
        xticks=listnbrWeights, xrotation=45)
    savefig("ukp-threads-err-" * string(n) * "-" * string(o) * "-T" * string(nT))

    plot(listnbrWeights, av_time,
        marker=:circle, lw=2, color=:red, legend=false,
        title="$n variables | $o objectives | $nT thread(s)",
        xlabel="Number of directions N",
        ylabel="avg. elapsed time (s)",
        xticks=listnbrWeights, xrotation=45)
    savefig("ukp-threads-time-" * string(n) * "-" * string(o) * "-T" * string(nT))

end

close(fresults)
nothing

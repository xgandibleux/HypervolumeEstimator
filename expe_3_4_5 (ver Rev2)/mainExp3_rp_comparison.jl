#= EXPERIMENT — UKP reference point comparison:

  - given a value for n (variables) and o (objectives)
    - warm-up: JIT compilation on a minimal instance (not timed)
    - generate nInstances instances randomly
    - for each instance:
        - compute Y_N : exact nondominated set (Tamby-Vanderpooten via MOA)
        - compute rp0    = zeros(o)              -> measure H0
        - compute rp_LB  = reference_point_LB() -> measure H_LB
        - for each value in listnbrWeights, for each rp choice:
            - estimate H with Hrevised3 (L_threaded, Threads.@threads)
            - collect normalized value, CI 95%, relative error, CPU time
    - compute across instances:
        - average H_estimated_normalized, CI, relative error, CPU time
        - for both rp choices
  - report:
    - comparison table: rp=0 vs rp=LB for each N
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

n = 50
o = 3

nInstances     = 10
listnbrWeights = [100, 500, 1000, 1500, 2000, 5000, 10000]
nWeights       = 7

println("  number of variables  : ", n)
println("  number of objectives : ", o)
println("  number of weights    : ", listnbrWeights)
println("  number of instances  : ", nInstances)
println("  solver MIP invoked   : ", solver)
println("  resolution strategy  : ", solver_fn)

# results matrices for both rp choices
res_norm_rp0  = zeros(Float64, nInstances, nWeights)
res_norm_rpLB = zeros(Float64, nInstances, nWeights)
res_err_rp0   = zeros(Float64, nInstances, nWeights)
res_err_rpLB  = zeros(Float64, nInstances, nWeights)
res_CIlo_rp0  = zeros(Float64, nInstances, nWeights)
res_CIhi_rp0  = zeros(Float64, nInstances, nWeights)
res_CIlo_rpLB = zeros(Float64, nInstances, nWeights)
res_CIhi_rpLB = zeros(Float64, nInstances, nWeights)
res_t_rp0     = zeros(Float64, nInstances, nWeights)
res_t_rpLB    = zeros(Float64, nInstances, nWeights)

instanceName = "ukp-rp-" * string(n) * "-" * string(o)
fresults     = open("ukp-tableResultsRPComparison.res", "a")

# =============================================================================
# Warm-up
println("Warm-up (JIT compilation)...")
let inst_w = generate_MO01UKP(2, 2)
    rp_w   = reference_point_LB(inst_w)
    S_w, _ = solve_MO01UKP(solver, inst_w)
    if length(S_w) > 0
        compute_Hmeasure(S_w, inst_w.o, zeros(Int, inst_w.o))
        compute_Hmeasure(S_w, inst_w.o, rp_w)
        Hrevised3(1.0, solver, inst_w, zeros(Int, inst_w.o), 10, solver_fn)
        Hrevised3(1.0, solver, inst_w, rp_w,                 10, solver_fn)
    end
end
println("Warm-up done.")
println("-"^80)

open(instanceName * ".res", "w") do ioAll
    write(ioAll, string(instanceName, "\n\n"))

    for iInstance in 1:nInstances

        println("\n---- instance $iInstance -------------------------------")

        # ==== INSTANCE ===========================================================
        inst = generate_MO01UKP(n, o)
        rp0  = zeros(Int, o)
        rpLB = reference_point_LB(inst)
        println("  rp0  = ", rp0)
        println("  rpLB = ", rpLB)

        # ==== EXACT ==============================================================
        println("\nCompute S = Y_N...")
        S, cardS = solve_MO01UKP(solver, inst)
        println("  |S| = ", cardS)
        write(ioAll, string("|S| = ", cardS, "\n"))

        H0  = compute_Hmeasure(S, inst.o, rp0)
        HLB = compute_Hmeasure(S, inst.o, rpLB)
        @printf("  H(S) with rp0  = %1.6e\n", H0)
        @printf("  H(S) with rpLB = %1.6e\n", HLB)
        write(ioAll, string("rp0  = ", rp0,  "\n"))
        write(ioAll, string("rpLB = ", rpLB, "\n"))
        write(ioAll, string("H(rp0)  = ", H0,  "\n"))
        write(ioAll, string("H(rpLB) = ", HLB, "\n\n"))

        # ==== ESTIMATION =========================================================
        println("\nEstimation with rp0 and rpLB:")

        for iWeight in 1:nWeights
            N = listnbrWeights[iWeight]

            # --- rp = zeros ---
            t0 = time()
            H̃0, (n0, CIlo0, CIhi0) = Hrevised3(H0, solver, inst, rp0, N, solver_fn)
            res_t_rp0[iInstance, iWeight]     = round(time() - t0, digits=2)
            res_norm_rp0[iInstance, iWeight]  = n0
            res_err_rp0[iInstance, iWeight]   = abs(1.0 - n0)
            res_CIlo_rp0[iInstance, iWeight]  = CIlo0
            res_CIhi_rp0[iInstance, iWeight]  = CIhi0

            # --- rp = LB ---
            tLB = time()
            H̃LB, (nLB, CIloLB, CIhiLB) = Hrevised3(HLB, solver, inst, rpLB, N, solver_fn)
            res_t_rpLB[iInstance, iWeight]     = round(time() - tLB, digits=2)
            res_norm_rpLB[iInstance, iWeight]  = nLB
            res_err_rpLB[iInstance, iWeight]   = abs(1.0 - nLB)
            res_CIlo_rpLB[iInstance, iWeight]  = CIloLB
            res_CIhi_rpLB[iInstance, iWeight]  = CIhiLB

            @printf("  N=%5d | rp0: norm=%.5f CI=[%.5f,%.5f] err=%.5f | rpLB: norm=%.5f CI=[%.5f,%.5f] err=%.5f\n",
                N,
                n0,  CIlo0,  CIhi0,  abs(1.0-n0),
                nLB, CIloLB, CIhiLB, abs(1.0-nLB))
        end
    end

    write(ioAll, "\n")

    # ==== AVERAGES ===============================================================
    av_norm_rp0  = [average_value(res_norm_rp0[:,i])  for i in 1:nWeights]
    av_norm_rpLB = [average_value(res_norm_rpLB[:,i]) for i in 1:nWeights]
    av_err_rp0   = [average_value(res_err_rp0[:,i])   for i in 1:nWeights]
    av_err_rpLB  = [average_value(res_err_rpLB[:,i])  for i in 1:nWeights]
    av_CIw_rp0   = [average_value(res_CIhi_rp0[:,i] .- res_CIlo_rp0[:,i])  for i in 1:nWeights]
    av_CIw_rpLB  = [average_value(res_CIhi_rpLB[:,i] .- res_CIlo_rpLB[:,i]) for i in 1:nWeights]
    av_t_rp0     = [average_value(res_t_rp0[:,i])     for i in 1:nWeights]
    av_t_rpLB    = [average_value(res_t_rpLB[:,i])    for i in 1:nWeights]

    println("\n", "="^80)
    println("Summary — average over $nInstances instances")
    println("="^80)
    @printf("%-8s | %-30s | %-30s\n", "N", "rp = zeros(o)", "rp = LB")
    @printf("%-8s | %-10s %-10s %-8s | %-10s %-10s %-8s\n",
            "", "norm", "CI width", "rel.err", "norm", "CI width", "rel.err")
    println("-"^80)
    for i in 1:nWeights
        @printf("%-8d | %-10.5f %-10.5f %-8.5f | %-10.5f %-10.5f %-8.5f\n",
            listnbrWeights[i],
            av_norm_rp0[i],  av_CIw_rp0[i],  av_err_rp0[i],
            av_norm_rpLB[i], av_CIw_rpLB[i], av_err_rpLB[i])
    end

    # ==== WRITE TABLE ============================================================
    print(fresults, " n  &  o  &  N  &  norm_rp0  &  CIwidth_rp0  &  err_rp0  &  norm_rpLB  &  CIwidth_rpLB  &  err_rpLB \n")
    for i in 1:nWeights
        print(fresults, " $n  &  $o  &  $(listnbrWeights[i])  & ")
        @printf(fresults, " %.6f  & %.6e  & %.6f  & ", av_norm_rp0[i], av_CIw_rp0[i], av_err_rp0[i])
        @printf(fresults, " %.6f  & %.6e  & %.6f \\\\\n", av_norm_rpLB[i], av_CIw_rpLB[i], av_err_rpLB[i])
    end

    # ==== PLOTS ==================================================================
    plot(listnbrWeights, av_err_rp0,
        label="rp = zeros", marker=:circle, lw=2, color=:blue)
    plot!(listnbrWeights, av_err_rpLB,
        label="rp = LB", marker=:square, lw=2, color=:red,
        title="$n variables | $o objectives — relative error comparison",
        xlabel="Number of weight vectors/iterations",
        ylabel="avg. relative error on H estimated normalized",
        xticks=listnbrWeights, xrotation=45, legend=:topright)
    savefig("ukp-rp-comparison-err-" * string(n) * "-" * string(o))

    plot(listnbrWeights, av_CIw_rp0,
        label="rp = zeros", marker=:circle, lw=2, color=:blue)
    plot!(listnbrWeights, av_CIw_rpLB,
        label="rp = LB", marker=:square, lw=2, color=:red,
        title="$n variables | $o objectives — CI width comparison",
        xlabel="Number of weight vectors/iterations",
        ylabel="avg. CI 95% width",
        xticks=listnbrWeights, xrotation=45, legend=:topright)
    savefig("ukp-rp-comparison-CI-" * string(n) * "-" * string(o))

end

close(fresults)
nothing

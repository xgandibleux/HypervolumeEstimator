#= EXPERIMENT 4 (second part) — orchestration scheme x Gurobi threading policy (-revT)

  Four execution modes (moUKP, n variables, o objectives, 10 instances):
    A : 1 chunk,   1 thread,  Gurobi constrained to 1 thread
    B : 1 chunk,   1 thread,  Gurobi free
    C : 50 chunks, 8 threads, Gurobi constrained to 1 thread
    D : 50 chunks, 8 threads, Gurobi free

  The modes use the SAME instances (same seed, same warm-up as the first part of
  Experiment 4) and the SAME sampled directions: only the elapsed time differs.
  The Julia thread count selects the modes executed by a run:
    --threads 1 -> modes A and B        --threads 8 -> modes C and D
  Run both (any order). When the four modes are available, the script writes the
  table of average elapsed times and of the speedups t_A/t_B, t_A/t_C, t_D/t_C.

  Usage (shell, from the folder containing this file; paper: 50 3):
    julia --threads 1 mainExp4_2x2_modes-revT.jl 50 3
    julia --threads 8 mainExp4_2x2_modes-revT.jl 50 3
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
using DelimitedFiles

Random.seed!(1234)

include("src/common-revT.jl")
include("src/instance_UKP.jl")
include("src/solve_UKP.jl")
include("src/files.jl")
include("src/estimHyperVol_UKP-revT.jl")   # Hrevised3
include("src/modes-revT.jl")                # L_modes
include("src/analyze.jl")

println("-"^80)

solver = Gurobi.Optimizer
n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 50
o = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3

nInstances     = 10
listnbrWeights = [100, 500, 1000, 1500, 2000, 5000, 10000]
nWeights       = length(listnbrWeights)
nT             = Threads.nthreads()

# mode => (orchsize, gurobi_free); orchsize <= 0 means a single chunk
MODES = Dict("A" => (0, false), "B" => (0, true),
             "C" => (CHUNKSIZE, false), "D" => (CHUNKSIZE, true))
modes = nT == 1 ? ["A", "B"] : ["C", "D"]

println("  variables / objectives : ", n, " / ", o)
println("  instances              : ", nInstances)
println("  directions N           : ", listnbrWeights)
println("  julia threads          : ", nT)
println("  modes of this run      : ", modes)
nT in (1, 8) || println("  WARNING: the paper uses --threads 1 (A,B) and --threads 8 (C,D).")

makeL(m) = (s, inst, rp, N) -> L_modes(s, inst, rp, N;
                                       orchsize = MODES[m][1], gurobi_free = MODES[m][2])

res_norm = Dict(m => zeros(nInstances, nWeights) for m in modes)
res_time = Dict(m => zeros(nInstances, nWeights) for m in modes)

# =============================================================================
# Warm-up (same single call to generate_MO01UKP(2,2) as the first part of Exp. 4,
# so that the random instances below are the same)
println("Warm-up (JIT compilation)...")
let inst_w = generate_MO01UKP(2, 2)
    rp_w   = reference_point_LB(inst_w)
    S_w, _ = solve_MO01UKP(solver, inst_w)
    if length(S_w) > 0
        compute_Hmeasure(S_w, inst_w.o, rp_w)
        for m in modes
            Hrevised3(1.0, solver, inst_w, rp_w, 10, makeL(m))
        end
    end
end
println("Warm-up done.")
println("-"^80)

for iInstance in 1:nInstances
    println("\n---- instance $iInstance -------------------------------")
    inst = generate_MO01UKP(n, o)
    rp   = reference_point_LB(inst)
    S, cardS = solve_MO01UKP(solver, inst)
    Hmeasure = compute_Hmeasure(S, inst.o, rp)
    @printf("  rp = %s | |S| = %d | H(S) = %1.6e\n", string(rp), cardS, Hmeasure)

    for iWeight in 1:nWeights
        N = listnbrWeights[iWeight]
        for m in modes
            t0 = time()
            _, (H_norm, CIlo, CIhi) = Hrevised3(Hmeasure, solver, inst, rp, N, makeL(m))
            el = round(time() - t0, digits=2)
            res_norm[m][iInstance, iWeight] = H_norm
            res_time[m][iInstance, iWeight] = el
            @printf("  N=%5d | mode %s | norm=%.5f t=%.2fs\n", N, m, H_norm, el)
        end
    end
end

# =============================================================================
tag(m) = string(n, "-", o, "-", m)
av_time = Dict(m => [average_value(res_time[m][:, i]) for i in 1:nWeights] for m in modes)

println("\n", "="^80)
println("Summary — $n variables | $o objectives | average elapsed time (s) over $nInstances instances")
println("="^80)
@printf("%-8s", "N"); for m in modes @printf(" | mode %-6s", m) end; println()
for i in 1:nWeights
    @printf("%-8d", listnbrWeights[i])
    for m in modes @printf(" | %-11.2f", av_time[m][i]) end
    println()
end

# per-mode dumps (full precision) -> used for the cross-checks and the final table
for m in modes
    writedlm("ukp-modes-norm-" * tag(m) * ".txt", res_norm[m])
    writedlm("ukp-modes-time-" * tag(m) * ".txt", av_time[m])
end

# --- cross-check: all modes must give (numerically) the same estimates --------
let files = [(m, "ukp-modes-norm-" * tag(m) * ".txt") for m in ("A", "B", "C", "D")
             if isfile("ukp-modes-norm-" * tag(m) * ".txt")]
    ref = nothing
    for (m, f) in files
        M = readdlm(f)
        ref === nothing && (ref = (m, M); continue)
        @printf("CHECK estimates mode %s vs mode %s : max abs. difference = %.3e\n",
                m, ref[1], maximum(abs.(M .- ref[2])))
    end
    # mode C must also coincide with the first part of Experiment 4 (T = 8)
    f48 = "ukp-verif-" * string(n) * "-" * string(o) * "-T8.txt"
    if isfile(f48) && isfile("ukp-modes-norm-" * tag("C") * ".txt")
        d = maximum(abs.(readdlm("ukp-modes-norm-" * tag("C") * ".txt") .- readdlm(f48)))
        @printf("CHECK mode C vs Experiment 4 (50 chunks, T=8) : max abs. difference = %.3e\n", d)
    end
end

# --- final table when the four modes are available -----------------------------
if all(isfile("ukp-modes-time-" * tag(m) * ".txt") for m in ("A", "B", "C", "D"))
    t = Dict(m => vec(readdlm("ukp-modes-time-" * tag(m) * ".txt")) for m in ("A", "B", "C", "D"))
    open("ukp-modes-table-" * string(n) * "-" * string(o) * ".res", "w") do io
        println(io, "N & A & B & C & D & tA/tB & tA/tC & tD/tC \\\\")
        for i in 1:nWeights
            @printf(io, "%d & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\\n",
                    listnbrWeights[i], t["A"][i], t["B"][i], t["C"][i], t["D"][i],
                    t["A"][i] / t["B"][i], t["A"][i] / t["C"][i], t["D"][i] / t["C"][i])
        end
    end
    println("\nTable written: ukp-modes-table-$(n)-$(o).res")
end
nothing

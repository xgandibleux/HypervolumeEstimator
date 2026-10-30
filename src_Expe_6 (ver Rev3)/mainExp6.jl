#= EXPERIMENT 6 — hypervolume estimation on a continuous multi-objective LP
  (weighted simplex), reference point = origin, sweep over the number of
  objectives d, for unit parameters (a = b = c = 1) and for random integer
  parameters in [1, 100]:

      max z_k = a_k x_k  (k = 1..d)   s.t.  sum_i b_i x_i <= c,  x >= 0

  H is known in closed form: H = prod_k (a_k c / b_k) / d!   (no call to HV)

  - warm-up: JIT compilation on a minimal instance (not timed)
  - for each parameter mode in listmodes ("unit", "rand") and each d in listd:
      - sanity check: LP value vs closed form L(ψ) on random directions
      - generate nInstances instances (a = b = c = 1, or integers in [1,100])
      - for each instance:
          - compute H exactly by formula
          - for each value in listnbrWeights:
              - estimate H with Hrevised3 (L_threaded, Threads.@threads)
              - collect normalized value, CI 95%, relative error, CPU time
      - average across instances
  - report: per mode, summary table over (d, N), LaTeX file, and plots
    (one curve per d, log-log axes)
=#

using Printf
using Random
using Base.Threads
using JuMP, Gurobi
using Distributions
using SpecialFunctions
using HypothesisTests
using Statistics
using Plots

Random.seed!(1234)

include("src/common.jl")
include("src/instance_LP.jl")
include("src/estimHyperVol_LP.jl")
include("src/analyze.jl")

println("-"^80)

# =============================================================================
println("Setup the parameters...")

# One Gurobi environment per thread, created once: avoids opening (and printing
# the license message of) a new environment for every model.
# Any other LP solver supported by JuMP works too (e.g. solver = HiGHS.Optimizer).
const GRB_ENVS = [Gurobi.Env() for _ in 1:Threads.maxthreadid()]
solver    = () -> Gurobi.Optimizer(GRB_ENVS[Threads.threadid()])
solver_fn = L_threaded            # or L_analytic_threaded (closed form, no solver)

listd = [2, 3, 5, 8]              # numbers of objectives (= numbers of variables)

listmodes     = ["unit"]          # add "rand" to also run random parameters  # "unit": a = b = c = 1
                                  # "rand": a_k, b_k, c integers drawn in [1, maxval]
maxval        = 100

nInstances     = 20                # 20 replications, as in the paper
listnbrWeights = [100, 500, 1000, 1500, 2000, 5000, 10000, 50000, 100000]
nWeights       = length(listnbrWeights)

println("  number of objectives : ", listd)
println("  parameter modes      : ", listmodes, "  (rand: integers in [1,$maxval])")
println("  number of weights    : ", listnbrWeights)
println("  number of instances  : ", nInstances)
println("  solver invoked       : Gurobi.Optimizer (one Env per thread)")
println("  resolution strategy  : ", solver_fn)
println("  reference point      : origin")


# =============================================================================
# Sanity check: LP value vs closed form of L on random directions
function sanity_check(d, random_params)
    inst_c = generate_LP(d; random=random_params, maxval=maxval)
    rng_c  = MersenneTwister(1)
    psi_c  = zeros(Float64, d)
    lam_c  = zeros(Float64, d)
    maxdiff = 0.0
    for _ in 1:20
        ψ!(psi_c, rng_c)
        λ!(lam_c, psi_c)
        m = Model(solver)
        set_silent(m)
        @variable(m, x[1:d] ≥ 0)
        @constraint(m, sum(inst_c.b[i] * x[i] for i in 1:d) ≤ inst_c.c)
        @variable(m, α ≥ 0)
        @constraint(m, [k=1:d], α ≤ lam_c[k] * inst_c.a[k] * x[k])
        @objective(m, Max, α)
        optimize!(m)
        maxdiff = max(maxdiff, abs(objective_value(m) - L_analytic(inst_c, psi_c)) / L_analytic(inst_c, psi_c))
    end
    @printf("  max relative |L_LP - L_analytic| over 20 directions = %.3e\n", maxdiff)
end


# =============================================================================
# Experiment for one value of d: returns the averages over the instances
function run_d(d::Int, mode::String)
    random_params = (mode == "rand")

    res_norm = zeros(Float64, nInstances, nWeights)
    res_err  = zeros(Float64, nInstances, nWeights)
    res_CIlo = zeros(Float64, nInstances, nWeights)
    res_CIhi = zeros(Float64, nInstances, nWeights)
    res_t    = zeros(Float64, nInstances, nWeights)

    println("\n", "="^80)
    println("mode = $mode | d = $d objectives")
    println("="^80)
    sanity_check(d, random_params)

    open("expe6-" * mode * "-" * string(d) * "obj.res", "w") do ioAll
        write(ioAll, string("expe6-", mode, "-", d, "obj\n\n"))

        for iInstance in 1:nInstances

            println("\n---- $mode | d = $d | instance $iInstance -------------------")

            inst = generate_LP(d; random=random_params, maxval=maxval)
            rp   = zeros(Int, d)
            println("  a = ", inst.a, "  b = ", inst.b, "  c = ", inst.c)
            write(ioAll, string("a = ", inst.a, "  b = ", inst.b, "  c = ", inst.c, "\n"))

            H = exact_hypervolume(inst)
            @printf("  H exact = %1.6e\n", H)
            write(ioAll, string("rp    = ", rp, "\n"))
            write(ioAll, string("H(rp) = ", H, "\n\n"))

            # a different seed range per instance: direction i uses seed + i, so
            # the ranges of two instances must not overlap (N <= 10^6)
            seed = 1234 + 1_000_000 * iInstance

            for iWeight in 1:nWeights
                N = listnbrWeights[iWeight]

                t0 = time()
                H̃, (nrm, CIlo, CIhi) = Hrevised3(H, solver, inst, rp, N, solver_fn, seed)
                res_t[iInstance, iWeight]    = round(time() - t0, digits=2)
                res_norm[iInstance, iWeight] = nrm
                res_err[iInstance, iWeight]  = abs(1.0 - nrm)
                res_CIlo[iInstance, iWeight] = CIlo
                res_CIhi[iInstance, iWeight] = CIhi

                @printf("  N=%5d | norm=%.5f CI=[%.5f,%.5f] err=%.5f\n",
                    N, nrm, CIlo, CIhi, abs(1.0 - nrm))
            end
        end
    end

    return (norm = [average_value(res_norm[:,i]) for i in 1:nWeights],
            err  = [average_value(res_err[:,i])  for i in 1:nWeights],
            CIw  = [average_value(res_CIhi[:,i] .- res_CIlo[:,i]) for i in 1:nWeights],
            t    = [average_value(res_t[:,i])    for i in 1:nWeights])
end


# =============================================================================
# Warm-up
println("Warm-up (JIT compilation)...")
let inst_w = generate_LP(2)
    rp_w = zeros(Int, inst_w.d)
    Hrevised3(exact_hypervolume(inst_w), solver, inst_w, rp_w, 10, solver_fn)
end
println("Warm-up done.")
println("-"^80)

# =============================================================================
# Sweep over the parameter modes and over d
results = Dict{Tuple{String,Int}, Any}()
for mode in listmodes, d in listd
    results[(mode, d)] = run_d(d, mode)
end

# =============================================================================
# Summary tables, LaTeX files and plots (one set per mode)
for mode in listmodes

    println("\n", "="^80)
    println("Summary [$mode] — average over $nInstances instances (rp = origin)")
    println("="^80)
    @printf("%-4s %-8s | %-10s %-10s %-10s %-8s\n", "d", "N", "norm", "CI width", "rel.err", "time(s)")
    println("-"^80)
    for d in listd
        r = results[(mode, d)]
        for i in 1:nWeights
            @printf("%-4d %-8d | %-10.5f %-10.5f %-10.5f %-8.2f\n",
                d, listnbrWeights[i], r.norm[i], r.CIw[i], r.err[i], r.t[i])
        end
        println("-"^80)
    end

    # LaTeX-ready table (overwritten at each run)
    open("expe6-" * mode * "-tableResults.res", "w") do fresults
        print(fresults, " d  &  N  &  norm  &  CIwidth  &  err  &  time \\\\\n")
        for d in listd
            r = results[(mode, d)]
            for i in 1:nWeights
                @printf(fresults, " %d  &  %d  &  %.6f  &  %.6e  &  %.6f  &  %.2f \\\\\n",
                    d, listnbrWeights[i], r.norm[i], r.CIw[i], r.err[i], r.t[i])
            end
        end
    end

    # plots: one curve per d, log-log axes (a 1/sqrt(N) decay is a straight line)
    p1 = plot(title="relative error vs N [$mode]", xlabel="Number of weight vectors/iterations",
              ylabel="avg. relative error on H estimated normalized",
              xscale=:log10, yscale=:log10, legend=:topright)
    p2 = plot(title="CI width vs N [$mode]", xlabel="Number of weight vectors/iterations",
              ylabel="avg. CI 95% width",
              xscale=:log10, yscale=:log10, legend=:topright)
    for d in listd
        plot!(p1, listnbrWeights, max.(results[(mode, d)].err, 1e-6), label="d = $d", marker=:circle, lw=2)
        plot!(p2, listnbrWeights, results[(mode, d)].CIw, label="d = $d", marker=:circle, lw=2)
    end
    savefig(p1, "expe6-" * mode * "-err")
    savefig(p2, "expe6-" * mode * "-CI")
end

nothing

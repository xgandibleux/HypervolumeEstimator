# =============================================================================
# Estimation of the hypervolume of a multi-objective continuous linear problem
# without knowing its set of nondominated points (Experiment 6: weighted
# simplex, see instance_LP.jl).
#
# Derived from the UKP estimator: the subproblem for each direction is now an
# LP (continuous variables). Parallelisation strategy unchanged:
#   - one model per thread, sequential reuse within each thread
#   - ψ! and λ! in-place with pre-allocated buffers; the RNG is reseeded with
#     (seed + direction index) so results do not depend on threads/scheduling
#   - Float64 coefficients pre-computed once before the parallel loop
#   - the RNG seed is an argument, so that different instances/repetitions
#     use different directions


# ------------------------------------------------------------
"""
    L_threaded(solver, inst::Instance_LP, rp, N, seed=1234)

Compute N values of L(S,r_*,ψ) — parallelised version.

The N directions are split into T contiguous blocks (T = number of threads).
Each thread builds its own LP once, then for each direction adds the d
Tchebychev constraints on α, solves, and deletes the constraints.

Returns a Vector{Float64} of length N.
"""
function L_threaded(solver, inst::Instance_LP,
                    rp::Vector{Int64}, N::Int, seed::Int=1234)

    d = inst.d

    af  = Float64.(inst.a)
    bf  = Float64.(inst.b)
    cf  = Float64(inst.c)
    rpf = Float64.(rp)

    listL = Vector{Float64}(undef, N)

    nt     = Threads.nthreads()
    chunks = collect(Iterators.partition(1:N, ceil(Int, N / nt)))

    Threads.@threads for chunk in chunks

        # buffers local to the block; the RNG is reseeded for each direction
        # (see below), so that direction i only depends on (seed, i): results
        # are reproducible whatever the number of threads and the scheduling
        rng = MersenneTwister()
        pb  = zeros(Float64, d)
        lb  = zeros(Float64, d)

        model = Model(solver)
        set_silent(model)
        @variable(model, x[1:d] ≥ 0)
        @constraint(model, sum(bf[i] * x[i] for i in 1:d) ≤ cf)
        @expression(model, z[k=1:d], af[k] * x[k])
        @variable(model, α ≥ 0)
        @objective(model, Max, α)

        for i in chunk
            Random.seed!(rng, seed + i)
            lw = λ!(lb, ψ!(pb, rng))
            @constraint(model, con[k=1:d], α ≤ -lw[k] * (rpf[k] - z[k]))
            JuMP.optimize!(model)
            @assert is_solved_and_feasible(model) "Error: optimal solution not found"
            listL[i] = objective_value(model)
            for k = 1:d
                delete(model, con[k])
            end
            unregister(model, :con)
        end

    end

    return listL
end


# ------------------------------------------------------------
"""
    L_analytic_threaded(solver, inst::Instance_LP, rp, N, seed=1234)

Same signature as L_threaded, but L is computed from the closed form
(no solver call). Only valid with rp = zeros. Useful to separate the sampling
error of the estimator from any modelling error in the LP.
"""
function L_analytic_threaded(solver, inst::Instance_LP,
                             rp::Vector{Int64}, N::Int, seed::Int=1234)

    @assert all(rp .== 0) "L_analytic_threaded is only valid with rp = zeros"

    d     = inst.d
    rng   = MersenneTwister()
    psi   = zeros(Float64, d)
    listL = Vector{Float64}(undef, N)

    for i in 1:N
        Random.seed!(rng, seed + i)     # same directions as L_threaded
        ψ!(psi, rng)
        listL[i] = L_analytic(inst, psi)
    end

    return listL
end


# ------------------------------------------------------------
"""
    Hrevised3(Hexact, solver, inst::Instance_LP, rp, numberOfWeights, solver_fn, seed=1234)

Compute H(S,r_*) — main estimator used in the experiment.

solver_fn selects the resolution strategy: L_threaded (LP) or
L_analytic_threaded (closed form).
Returns H_estimated and a tuple (H_estimated_normalized, CI_low, CI_high)
where normalization is with respect to Hexact and the CI is at 95%.
"""
function Hrevised3(Hexact, solver, inst::Instance_LP, rp, numberOfWeights, solver_fn, seed::Int=1234)

    o = inst.d

    listL       = solver_fn(solver, inst, rp, numberOfWeights, seed)
    E           = sum(x -> x^o, listL) / numberOfWeights
    H_estimated = 1/o * (2 * π^(o/2)) / (gamma(o/2) * 2^o) * E

    listL_weighted            = 1/o * (2 * π^(o/2)) / (gamma(o/2) * 2^o) .* (listL .^ o)
    listL_weighted_normalized = listL_weighted ./ Hexact
    H_estimated_normalized    = average_value(listL_weighted_normalized)

    CIlow, CIHigh = confint(OneSampleTTest(listL_weighted_normalized), level=0.95, tail=:both)

    return H_estimated, (H_estimated_normalized, CIlow, CIHigh)
end

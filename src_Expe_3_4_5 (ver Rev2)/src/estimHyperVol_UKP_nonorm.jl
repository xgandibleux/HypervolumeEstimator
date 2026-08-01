# =============================================================================
# Hypervolume estimator for large UKP instances — non-normalised variant.
#
# Hrevised3_nonorm does not require Hexact: it returns H_estimated (raw value)
# and a 95% CI computed on the raw weighted L^o values (intra-run).
#
# L_threaded_seeded is a variant of L_threaded that accepts an explicit seed
# for the MersenneTwister RNGs, allowing different runs to produce different
# directions on the same instance.
#
# ψ!, λ! are defined in common.jl — do not redefine them here.


# ------------------------------------------------------------
"""
    L_threaded_seeded(solver, inst::Instance_UKP, rp, N, seed)

Same as L_threaded but uses `seed` to initialise the per-thread RNGs,
so that different calls with different seeds produce different directions.
"""
function L_threaded_seeded(solver::DataType,
                            inst::Instance_UKP,
                            rp::Vector{Int64}, N::Int, seed::Int)

    o = inst.o
    n = inst.n

    pf  = Float64.(inst.p)
    wf  = Float64.(inst.w)
    rpf = Float64.(rp)

    listL  = Vector{Float64}(undef, N)
    nt     = Threads.nthreads()
    chunks = collect(Iterators.partition(1:N, ceil(Int, N / nt)))

    # seed varies per run : different directions for each run
    rngs     = [MersenneTwister(seed + t) for t in 1:Threads.maxthreadid()]
    psi_bufs = [zeros(Float64, o)         for _ in 1:Threads.maxthreadid()]
    lam_bufs = [zeros(Float64, o)         for _ in 1:Threads.maxthreadid()]

    Threads.@threads for chunk in chunks
        tid = Threads.threadid()
        rng = rngs[tid]
        pb  = psi_bufs[tid]
        lb  = lam_bufs[tid]

        model = Model(solver)
        set_silent(model)
        occursin("Gurobi", string(solver)) && set_attribute(model, "Threads", 1)
        occursin("Gurobi", string(solver)) && set_attribute(model, "MIPGap",  0.0)
        @variable(model, x[1:n], Bin)
        @constraint(model, sum(wf[i] * x[i] for i in 1:n) ≤ inst.c)
        @expression(model, z[k=1:o], sum(pf[k,j] * x[j] for j in 1:n))
        @variable(model, α ≥ 0)
        @objective(model, Max, α)

        for i in chunk
            lw = λ!(lb, ψ!(pb, rng))
            @constraint(model, con[k=1:o], α ≤ -lw[k] * (rpf[k] - z[k]))
            JuMP.optimize!(model)
            @assert is_solved_and_feasible(model) "Error: optimal solution not found"
            listL[i] = objective_value(model)
            for k = 1:o; delete(model, con[k]); end
            unregister(model, :con)
        end
    end

    return listL
end


# ------------------------------------------------------------
"""
    Hrevised3_nonorm(solver, inst::Instance_UKP, rp, numberOfWeights, seed)

Compute H(S,r_*) without normalisation — for use when Hexact is unavailable.
seed controls the RNG so that different runs on the same instance produce
different directions and thus different estimates.

Returns H_estimated and a tuple (mean_lw, CI_low, CI_high).
"""
function Hrevised3_nonorm(solver, inst::Instance_UKP, rp, numberOfWeights, seed::Int)

    o = inst.o

    listL          = L_threaded_seeded(solver, inst, rp, numberOfWeights, seed)
    coeff          = 1/o * (2 * π^(o/2)) / (gamma(o/2) * 2^o)
    H_estimated    = coeff * sum(x -> x^o, listL) / numberOfWeights
    listL_weighted = coeff .* (listL .^ o)

    CIlow, CIhigh = confint(OneSampleTTest(listL_weighted), level=0.95, tail=:both)

    return H_estimated, (mean(listL_weighted), CIlow, CIhigh)
end

# =============================================================================
# Estimation of the hypervolume for the multi-objective UFLP.
#
# The UFLP minimises all objectives. The estimator is designed for
# maximisation, so objectives are negated: z̃_k = -z_k.
# The reference point rp = (-UB_1, ..., -UB_o) is computed per instance
# via reference_point(inst) defined in instanceUFLP.jl.
#
# July 2025
# version parallelised and optimised:
#   - one model per thread, sequential reuse within each thread
#   - direction generation fused into the parallel loop (one RNG per thread)
#   - ψ! and λ! in-place with pre-allocated buffers per thread (see common.jl)
#   - Float64 coefficients pre-computed once before the parallel loop
#   - sum without intermediate vector allocation


# ------------------------------------------------------------
"""
    L_threaded_UFLP(solver, inst::Instance_UFLP, rp, N)

Compute N values of L(S,r_*,ψ) for the multi-objective UFLP — parallelised.

Objectives are negated (minimisation → maximisation).
The JuMP model is built once per thread; the o Tchebychev constraints on α
are added, the model is solved, then deleted and unregistered at each direction.
"""
function L_threaded_UFLP(solver::DataType, inst::Instance_UFLP,
                          rp::Vector{Int64}, N::Int)

    nI = inst.nI
    nJ = inst.nJ
    o  = inst.o

    # Float64 coefficients pre-computed once — avoids implicit Int->Float64
    # conversion inside JuMP at every model build
    rf  = [Float64.(inst.r[k]) for k in 1:o]
    cf  = [Float64.(inst.c[k]) for k in 1:o]
    rpf = Float64.(rp)

    listL = Vector{Float64}(undef, N)

    nt     = Threads.nthreads()
    chunks = collect(Iterators.partition(1:N, ceil(Int, N / nt)))

    rngs     = [MersenneTwister(1234 + t) for t in 1:Threads.maxthreadid()]
    psi_bufs = [zeros(Float64, o)         for _ in 1:Threads.maxthreadid()]
    lam_bufs = [zeros(Float64, o)         for _ in 1:Threads.maxthreadid()]

    Threads.@threads for chunk in chunks

        tid = Threads.threadid()
        rng = rngs[tid]
        pb  = psi_bufs[tid]
        lb  = lam_bufs[tid]

        # build the UFLP model once per thread
        model = Model(solver)
        set_silent(model)
        occursin("Gurobi", string(solver)) && set_attribute(model, "Threads", 1)

        @variable(model, y[1:nJ], Bin)
        @variable(model, x[1:nI, 1:nJ], Bin)

        @constraint(model, cover[i=1:nI], sum(x[i,j] for j in 1:nJ) == 1)
        @constraint(model, link[i=1:nI, j=1:nJ], x[i,j] <= y[j])

        # negated objectives: z̃_k = -z_k  (minimisation → maximisation)
        @expression(model, z[k=1:o],
            -(sum(rf[k][j]*y[j] for j in 1:nJ) +
              sum(cf[k][i,j]*x[i,j] for i in 1:nI, j in 1:nJ)))

        @variable(model, α ≥ 0)
        @objective(model, Max, α)

        # sequential reuse within the thread's block
        for i in chunk
            lw = λ!(lb, ψ!(pb, rng))
            @constraint(model, con[k=1:o], α ≤ -lw[k] * (rpf[k] - z[k]))
            JuMP.optimize!(model)
            @assert is_solved_and_feasible(model) "Error: optimal solution not found"
            listL[i] = objective_value(model)
            for k in 1:o
                delete(model, con[k])
            end
            unregister(model, :con)
        end

    end

    return listL
end


# ------------------------------------------------------------
"""
    Hrevised3_UFLP(Hexact, solver, inst::Instance_UFLP, rp, numberOfWeights)

Compute H(S,r_*) for the multi-objective UFLP.
The reference point rp is passed explicitly (computed per instance in mainExp1_UFLP.jl
via reference_point(inst)).
Returns H_estimated and a tuple (H_estimated_normalized, CI_low, CI_high)
where normalization is with respect to Hexact and the CI is at 95%.
"""
function Hrevised3_UFLP(Hexact, solver, inst::Instance_UFLP,
                         rp::Vector{Int64}, numberOfWeights::Int)

    o = inst.o

    listL       = L_threaded_UFLP(solver, inst, rp, numberOfWeights)
    E           = sum(x -> x^o, listL) / numberOfWeights
    H_estimated = 1/o * (2 * π^(o/2)) / (gamma(o/2) * 2^o) * E

    listL_weighted            = 1/o * (2 * π^(o/2)) / (gamma(o/2) * 2^o) .* (listL .^ o)
    listL_weighted_normalized = listL_weighted ./ Hexact
    H_estimated_normalized    = average_value(listL_weighted_normalized)

    CIlow, CIHigh = confint(OneSampleTTest(listL_weighted_normalized),
                            level=0.95, tail=:both)

    return H_estimated, (H_estimated_normalized, CIlow, CIHigh)
end

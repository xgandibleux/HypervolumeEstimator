# =============================================================================
# Estimation of the hypervolume for the multi-objective UFLP.
#
# The UFLP minimises all objectives. The estimator is designed for
# maximisation, so objectives are negated: z̃_k = -z_k.
# The reference point rp = (-UB_1, ..., -UB_o) is computed per instance
# via reference_point(inst) defined in instanceUFLP.jl.
#
# July 2026
# version parallelised and optimised (-revT):
#   - directions partitioned into chunks of fixed size CHUNKSIZE,
#     independent of the number of threads; chunks dynamically assigned
#   - one model per chunk, sequential reuse within each chunk
#   - one RNG per chunk, seeded by the chunk index: the sampled directions
#     (hence the estimate) are identical for any value of --threads
#   - ψ! and λ! in-place with per-chunk pre-allocated buffers
#   - MIPGap set to 0.0: exact optimality of each L_i (no downward bias)
#   - Float64 coefficients pre-computed once before the parallel loop
#   - sum without intermediate vector allocation
#
# CHUNKSIZE, set_exact_and_singlethread!, ψ! and λ! are defined in
# common-revT.jl.


# ------------------------------------------------------------
"""
    L_threaded_UFLP(solver, inst::Instance_UFLP, rp, N)

Compute N values of L(S,r_*,ψ) for the multi-objective UFLP — parallelised.

Objectives are negated (minimisation → maximisation).

The N directions are split into contiguous chunks of fixed size
CHUNKSIZE, independent of the number of threads; the chunks are
dynamically assigned to the threads. Each chunk owns its own RNG (seeded
by the chunk index) and its own ψ/λ buffers, and builds its JuMP model
once (1 internal solver thread, MIPGap = 0); the o Tchebychev
constraints on α are added, the model is solved, then the constraints
are deleted and unregistered at each direction.

Since the partition and the seeds depend only on N and CHUNKSIZE — not
on the number of threads nor on the scheduling — the returned vector is
identical for any value of --threads.
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

    chunks = collect(Iterators.partition(1:N, CHUNKSIZE))

    Threads.@threads for chunk_id in eachindex(chunks)

        # one RNG and one pair of buffers per chunk: race-free and
        # independent of the task-to-thread scheduling
        rng = MersenneTwister(1234 + chunk_id)
        pb  = zeros(Float64, o)
        lb  = zeros(Float64, o)

        # build the UFLP model once per chunk
        model = Model(solver)
        set_silent(model)
        set_exact_and_singlethread!(model, solver)

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

        # sequential reuse within the chunk
        for i in chunks[chunk_id]
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

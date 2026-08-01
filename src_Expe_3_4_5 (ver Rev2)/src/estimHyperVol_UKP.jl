# =============================================================================
# Estimation of the hypervolume of an instance of a multi-objective 
# optimization problem without knowing its set of nondominated points.
#
# Example for a multi-objective 01 unidimensionnal knapsack problem.
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
    L_threaded(solver, inst::Instance_UKP, rp, N)

Compute N values of L(S,r_*,ψ) — parallelised version.

The N directions are split into T contiguous blocks (T = number of threads).
Each thread:
  - builds its own JuMP model once (Float64 coefficients, no repeated conversion)
  - owns its own MersenneTwister RNG and pre-allocated ψ/λ buffers
  - generates its directions and iterates sequentially over its block:
    the d Tchebychev constraints on α are added, the model is solved,
    then the constraints are deleted and unregistered before the next direction.

Returns a pre-allocated Vector{Float64} of length N.
"""
function L_threaded(solver::DataType,
                    inst::Instance_UKP,
                    rp::Vector{Int64}, N::Int)

    o = inst.o
    n = inst.n

    # Float64 coefficients pre-computed once — avoids implicit Int->Float64
    # conversion inside JuMP at every model build
    pf  = Float64.(inst.p)
    wf  = Float64.(inst.w)
    rpf = Float64.(rp)

    listL = Vector{Float64}(undef, N)

    nt     = Threads.nthreads()
    chunks = collect(Iterators.partition(1:N, ceil(Int, N / nt)))

    # one RNG and one pair of buffers per possible thread ID
    rngs     = [MersenneTwister(1234 + t) for t in 1:Threads.maxthreadid()]
    psi_bufs = [zeros(Float64, o)         for _ in 1:Threads.maxthreadid()]
    lam_bufs = [zeros(Float64, o)         for _ in 1:Threads.maxthreadid()]

    Threads.@threads for chunk in chunks

        tid = Threads.threadid()
        rng = rngs[tid]
        pb  = psi_bufs[tid]
        lb  = lam_bufs[tid]

        # each thread builds its own model once with Float64 coefficients
        model = Model(solver)
        set_silent(model)
        # limit Gurobi to 1 internal thread per model: parallelism is handled
        # by Threads.@threads; without this, multi-instance Gurobi oversubscribes
        #occursin("Gurobi", string(solver)) && set_attribute(model, "Threads", 1)
        @variable(model, x[1:n], Bin)
        @constraint(model, sum(wf[i] * x[i] for i in 1:n) ≤ inst.c)
        @expression(model, z[k=1:o], sum(pf[k,j] * x[j] for j in 1:n))
        @variable(model, α ≥ 0)
        @objective(model, Max, α)

        # sequential reuse within the thread's block
        for i in chunk
            lw = λ!(lb, ψ!(pb, rng))   # generate direction in-place
            @constraint(model, con[k=1:o], α ≤ -lw[k] * (rpf[k] - z[k]))
            JuMP.optimize!(model)
            @assert is_solved_and_feasible(model) "Error: optimal solution not found"
            listL[i] = objective_value(model)
            for k = 1:o
                delete(model, con[k])
            end
            unregister(model, :con)
        end

    end

    return listL
end


# ------------------------------------------------------------
"""
    Lbis_threaded(solver, inst::Instance_UKP, rp, N)

Compute N values of L(S,r_*,ψ) — parallelised version (Oscar's formulation).

Same threading and optimisation strategy as L_threaded. Each thread builds
its own model with an auxiliary variable rhs[k] = rp[k] - p*x, so the
Tchebychev constraints take the form α + λ_k * rhs[k] ≤ 0. Between
directions, only the coefficients of rhs are updated via
set_normalized_coefficient — no constraint deletion required.

Returns a pre-allocated Vector{Float64} of length N.
"""
function Lbis_threaded(solver::DataType,
                       inst::Instance_UKP,
                       rp::Vector{Int64}, N::Int)

    o = inst.o
    n = inst.n

    pf  = Float64.(inst.p)
    wf  = Float64.(inst.w)
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

        # each thread builds its own model once with Float64 coefficients
        model = Model(solver)
        set_silent(model)
        occursin("Gurobi", string(solver)) && set_attribute(model, "Threads", 1)
        @variable(model, x[1:n], Bin)
        @variable(model, rhs[1:o])
        @variable(model, α)
        @constraint(model, wf' * x <= inst.c)
        @constraint(model, rhs .== rpf .- pf * x)
        @objective(model, Max, 1.0 * α)
        @constraint(model, con[k in 1:o], α + 1.0 * rhs[k] <= 0)

        # sequential reuse: only coefficients change
        for i in chunk
            lw = λ!(lb, ψ!(pb, rng))
            set_normalized_coefficient.(con, rhs, lw)
            JuMP.optimize!(model)
            @assert is_solved_and_feasible(model) "Error: optimal solution not found"
            listL[i] = objective_value(model)
        end

    end

    return listL
end


# ------------------------------------------------------------
"""
    Hrevised3(Hexact, solver, inst::Instance_UKP, rp, numberOfWeights, solver_fn)

Compute H(S,r_*) — main estimator used in the experiment.

solver_fn selects the resolution strategy: pass L_threaded or Lbis_threaded.
Returns H_estimated and a tuple (H_estimated_normalized, CI_low, CI_high)
where normalization is with respect to Hexact and the CI is at 95%.
"""
function Hrevised3(Hexact, solver, inst::Instance_UKP, rp, numberOfWeights, solver_fn)

    o = inst.o

    listL       = solver_fn(solver, inst, rp, numberOfWeights)
    E           = sum(x -> x^o, listL) / numberOfWeights   # no intermediate vector
    H_estimated = 1/o * (2 * π^(o/2)) / (gamma(o/2) * 2^o) * E

    listL_weighted            = 1/o * (2 * π^(o/2)) / (gamma(o/2) * 2^o) .* (listL .^ o)
    listL_weighted_normalized = listL_weighted ./ Hexact
    H_estimated_normalized    = average_value(listL_weighted_normalized)

    CIlow, CIHigh = confint(OneSampleTTest(listL_weighted_normalized), level=0.95, tail=:both)

    return H_estimated, (H_estimated_normalized, CIlow, CIHigh)
end

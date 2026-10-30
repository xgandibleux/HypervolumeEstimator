# =============================================================================
# Experiment 4 (2x2 comparison) — orchestration scheme x Gurobi threading policy
#
#   A : 1 chunk,   1 thread,  Gurobi constrained to 1 thread
#   B : 1 chunk,   1 thread,  Gurobi free (default: automatic number of threads)
#   C : 50 chunks, 8 threads, Gurobi constrained to 1 thread
#   D : 50 chunks, 8 threads, Gurobi free
#
# (the number of Julia threads, 1 or 8, is given by `julia --threads`)
#
# The sampled directions are generated exactly as in L_threaded
# (one MersenneTwister per logical block of CHUNKSIZE directions, seeded by
# 1234 + block index), whatever the orchestration scheme: the four modes
# therefore estimate H from the SAME directions, and mode C returns the same
# values as L_threaded. Only the way the N MIPs are grouped into models
# (orchsize) and the solver threading policy differ, so that only the elapsed
# time is affected.
# =============================================================================

"""
    set_exact_threads!(model, solver; singlethread)

MIPGap = 0 (exact optimality of each L_i). If `singlethread`, the solver is
restricted to one internal thread; otherwise it is left free (solver default).
"""
function set_exact_threads!(model, solver::DataType; singlethread::Bool)
    if occursin("Gurobi", string(solver))
        singlethread && set_attribute(model, "Threads", 1)
        set_attribute(model, "MIPGap", 0.0)
    elseif occursin("HiGHS", string(solver))
        singlethread && set_attribute(model, "threads", 1)
        set_attribute(model, "mip_rel_gap", 0.0)
    end
    return model
end

"""
    L_modes(solver, inst, rp, N; orchsize, gurobi_free)

Compute N values of L(S,r_*,ψ).
  - orchsize   : number of directions per model/task (orchsize <= 0: a single
                 chunk containing the N directions)
  - gurobi_free: false -> 1 internal solver thread; true -> solver default
"""
function L_modes(solver::DataType, inst::Instance_UKP, rp::Vector{Int64}, N::Int;
                 orchsize::Int, gurobi_free::Bool)

    o = inst.o
    n = inst.n
    pf  = Float64.(inst.p)
    wf  = Float64.(inst.w)
    rpf = Float64.(rp)

    # --- directions: identical to L_threaded, independent of orchsize ---------
    Λ  = Matrix{Float64}(undef, N, o)
    pb = zeros(Float64, o)
    lb = zeros(Float64, o)
    for (bid, blk) in enumerate(Iterators.partition(1:N, CHUNKSIZE))
        rng = MersenneTwister(1234 + bid)
        for i in blk
            Λ[i, :] .= λ!(lb, ψ!(pb, rng))
        end
    end

    # --- orchestration: one model per chunk, chunks distributed on the threads
    listL  = Vector{Float64}(undef, N)
    chunks = collect(Iterators.partition(1:N, orchsize <= 0 ? N : orchsize))

    Threads.@threads for chunk_id in eachindex(chunks)
        model = Model(solver)
        set_silent(model)
        set_exact_threads!(model, solver; singlethread = !gurobi_free)
        @variable(model, x[1:n], Bin)
        @constraint(model, sum(wf[i] * x[i] for i in 1:n) ≤ inst.c)
        @expression(model, z[k=1:o], sum(pf[k,j] * x[j] for j in 1:n))
        @variable(model, α ≥ 0)
        @objective(model, Max, α)

        for i in chunks[chunk_id]
            @constraint(model, con[k=1:o], α ≤ -Λ[i, k] * (rpf[k] - z[k]))
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

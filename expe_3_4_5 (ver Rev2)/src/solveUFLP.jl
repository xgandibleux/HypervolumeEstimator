# =============================================================================
# Exact solver for the multi-objective UFLP using MultiObjectiveAlgorithms.jl
# (Tamby-Vanderpooten algorithm).

"""
    solve_UFLP(solver, inst::Instance_UFLP)

Compute the set of nondominated points Y_N of a multi-objective UFLP instance.
All objectives are minimised; MOA handles the multi-objective enumeration.
Returns (S, cardS) where S is a Vector of Int64 vectors (one per point).
"""
function solve_UFLP(solver, inst::Instance_UFLP)

    nI = inst.nI
    nJ = inst.nJ
    o  = inst.o

    mo_uflp = Model()
    set_silent(mo_uflp)

    @variable(mo_uflp, y[1:nJ], Bin)
    @variable(mo_uflp, x[1:nI, 1:nJ], Bin)

    @constraint(mo_uflp, cover[i=1:nI], sum(x[i,j] for j in 1:nJ) == 1)
    @constraint(mo_uflp, link[i=1:nI, j=1:nJ], x[i,j] <= y[j])

    # o objectives, generalised via loop
    @expression(mo_uflp, z[k=1:o],
        sum(inst.r[k][j]*y[j] for j in 1:nJ) +
        sum(inst.c[k][i,j]*x[i,j] for i in 1:nI, j in 1:nJ))

    @objective(mo_uflp, Min, [z[k] for k in 1:o])

    set_optimizer(mo_uflp, () -> MOA.Optimizer(solver))
    set_attribute(mo_uflp, MOA.Algorithm(), MOA.TambyVanderpooten())

    # setting the tolerance of the MIP solver to zero
    if occursin("Gurobi", string(solver))
        set_attribute(mo_uflp, "MIPGap", 0.0)
    else
        set_optimizer_attribute(mo_uflp, "primal_feasibility_tolerance", 1e-10)
        set_optimizer_attribute(mo_uflp, "dual_feasibility_tolerance",   1e-10)
        set_optimizer_attribute(mo_uflp, "mip_rel_gap",                  0.0)
        set_optimizer_attribute(mo_uflp, "mip_abs_gap",                  0.0)
    end

    optimize!(mo_uflp)
    @assert is_solved_and_feasible(mo_uflp) "Error: optimal solution not found"

    S = (Vector{Int64})[]
    for i in 1:result_count(mo_uflp)
        push!(S, round.(Int, objective_value(mo_uflp; result = i)))
    end

    return S, result_count(mo_uflp)
end

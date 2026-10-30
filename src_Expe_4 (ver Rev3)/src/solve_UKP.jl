"""
    solve_MO01UKP(solver, inst::Instance_UKP)

Compute the set of nondominated points Y_N of a multi-objective 01UKP
using the Tamby-Vanderpooten algorithm via MultiObjectiveAlgorithms.jl.
"""
function solve_MO01UKP(solver, inst::Instance_UKP)

    o = inst.o
    n = inst.n

    mo01UKP = Model()
    set_silent(mo01UKP)
    @variable(mo01UKP, x[1:n], Bin)
    @expression(mo01UKP, z[k=1:o], sum(inst.p[k,j] * x[j] for j in 1:n))
    @objective(mo01UKP, Max, [z[k] for k in 1:o])
    @constraint(mo01UKP, sum(inst.w[i] * x[i] for i in 1:n) ≤ inst.c)

    set_optimizer(mo01UKP, () -> MOA.Optimizer(solver))
    set_attribute(mo01UKP, MOA.Algorithm(), MOA.TambyVanderpooten())

    # setting the tolerance of the MIP solver to zero
    if occursin("Gurobi", string(solver))
        set_attribute(mo01UKP, "MIPGap", 0.0)
    else
        set_optimizer_attribute(mo01UKP, "primal_feasibility_tolerance", 1e-10)
        set_optimizer_attribute(mo01UKP, "dual_feasibility_tolerance",   1e-10)
        set_optimizer_attribute(mo01UKP, "mip_rel_gap",                  0.0)
        set_optimizer_attribute(mo01UKP, "mip_abs_gap",                  0.0)
    end

    optimize!(mo01UKP)
    @assert is_solved_and_feasible(mo01UKP) "Error: optimal solution not found"

    S = (Vector{Int64})[]
    for i in 1:result_count(mo01UKP)
        push!(S, round.(Int, objective_value(mo01UKP; result = i)))
    end

    return S, result_count(mo01UKP)
end

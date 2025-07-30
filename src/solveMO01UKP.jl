#using JuMP
#using GLPK, Gurobi, HiGHS
#import MultiObjectiveAlgorithms as MOA


# ------------------------------------------------------------
"""
    solve_MO01UKP(solver, p, w, c)

Solve an instance for the MO-01UKP 
"""
function solve_MO01UKP(solver, p, w, c)

    o,n = size(p)

    mo01UKP = Model( )
    set_silent(mo01UKP)
    @variable(mo01UKP, x[1:n], Bin)
    @expression(mo01UKP, z[k=1:o], sum(p[k,j] * x[j] for j in 1:n))  
    @objective(mo01UKP, Max, [z[k] for k=1:o])
    @constraint(mo01UKP, sum(w[i] * x[i] for i in 1:n) ≤ c)

    set_optimizer(mo01UKP, () -> MOA.Optimizer(solver))
    #if o==2
    #    set_attribute(mo01UKP, MOA.Algorithm(), MOA.EpsilonConstraint())
    #else
        set_attribute(mo01UKP, MOA.Algorithm(), MOA.TambyVanderpooten())
    #end
    #set_attribute(mo01UKP, MOI.TimeLimitSec(), 60)

    optimize!(mo01UKP)

    S = (Vector{Int64})[]
    for i in 1:result_count(mo01UKP)
        #print(i, ": x = ", round.(Int, value.(x; result = i)), " | ")
        #println("$i : z = ", round.(Int, objective_value(mo01UKP; result = i)))
        push!(S, round.(Int, objective_value(mo01UKP; result = i)) )
    end

    return S, result_count(mo01UKP)

end
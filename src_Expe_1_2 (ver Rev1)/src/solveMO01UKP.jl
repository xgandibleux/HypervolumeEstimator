#using JuMP
#using GLPK, Gurobi, HiGHS
#import MultiObjectiveAlgorithms as MOA


"""
    solve_01UKP(solver, p, w, c)

Compute the optimal solution of the single objective 01UKP
"""
function solve_01UKP(solver, p, w, c)

    n = length(p)
    binaryUKP = Model(solver)
    set_silent(binaryUKP)
    @variable(binaryUKP, x[1:n], Bin)
    @objective(binaryUKP, Max, sum(p[j] * x[j] for j in 1:n))
    @constraint(binaryUKP, sum(w[i] * x[i] for i in 1:n) ≤ c)

    #set_attribute(binaryUKP, MOI.TimeLimitSec(), 60)
    optimize!(binaryUKP)
    @assert is_solved_and_feasible(binaryUKP) "Error: optimal solution not found"

    zOpt = round(Int, objective_value(binaryUKP))
    #print("z = ", zOpt, " | ")
    #println("x = ", round.(Int, value.(x)))

    return zOpt, round.(Int, value.(x))
end



"""
    solve_MO01UKP(solver, p, w, c)

Compute the set of nondominated points of a multi-objective 01UKP
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
    set_attribute(mo01UKP, MOA.Algorithm(), MOA.TambyVanderpooten())
    #set_attribute(mo01UKP, MOI.TimeLimitSec(), 60)

    optimize!(mo01UKP)
    @assert is_solved_and_feasible(mo01UKP) "Error: optimal solution not found"

    S = (Vector{Int64})[]
    for i in 1:result_count(mo01UKP)
        #print(i, ": x = ", round.(Int, value.(x; result = i)), " | ")
        #println("$i : z = ", round.(Int, objective_value(mo01UKP; result = i)))
        push!(S, round.(Int, objective_value(mo01UKP; result = i)) )
    end

    return S, result_count(mo01UKP)

end


"""
    solve_scalarized01UKP(  
             solver::DataType, 
             p::Matrix{Int64},   w::Vector{Int64},   c::Int64,
             rp::Vector{Int64}, λ::Vector{Float64}
          )

Compute one nondominated points of a multi-objective 01UKP using the augmented weighted Tchebychev norm
"""
function solve_scalarized01UKP(  
             solver::DataType, 
             p::Matrix{Int64},   w::Vector{Int64},   c::Int64,
             rp::Vector{Int64}, λ::Vector{Float64}
          )

    
    d,n = size(p)       # number of objectives and number of variables
    
    # define a JuMP model
    model = Model(solver)  
    set_silent(model)

    # setup X according (1)
    @variable(model, x[1:n], Bin)                                    # the n binary variables of 01UKP
    @constraint(model, sum(w[i] * x[i] for i in 1:n) ≤ c)            # the constraint of 01UKP

    # declare the objective functions
    @expression(model, z[k=1:d], sum(p[k,j] * x[j] for j in 1:n))    # the d objectives of 01UKP

    # ------------------------------------------------------------------------------
    # Concerning the Chebychev part of the model

    @variable(model, α ≥ 0)        
    @objective(model, Min, α - sum(0.001 * z[k] for k=1:d))

    solve_time_sec = 0.0

    # add the d constraints for a given λ_ψ to the model
    @constraint(model, con[k=1:d], α ≥  λ[k] * (rp[k] -  z[k]) )
    
    #print(model)
    JuMP.optimize!(model)
    @assert is_solved_and_feasible(model) "Error: optimal solution not found"

    solve_time_sec += solve_time(model)
    
    #println("zOpt: ", objective_value(model))
    #print("xOpt: ", value.(x), " ", value(α))

    return round(Int, objective_value(model)), round.(Int, value.(x))
end

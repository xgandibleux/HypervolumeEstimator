using JuMP, GLPK, HiGHS, Gurobi, CPLEX

function L(  solver::DataType, 
             p::Matrix{Int64}, w::Vector{Int64}, c::Int64,
             rp::Vector{Int64}, λ_ψ::Vector{Vector{Float64}}
          )

    L = (Float64)[]  # a series of L(S,r*,ψ) values
    
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
    @objective(model, Max, α)

    for i in 1:length(λ_ψ)

        # add the d constraints for a given λ_ψ to the model
        @constraint(model, con[k=1:d], α ≤ - λ_ψ[i][k] * (rp[k] -  z[k]) )
        
        optimize!(model)
        @assert is_solved_and_feasible(model) "Error: optimal solution not found"

        push!(L,objective_value(model))
        
        for k=1:d
            delete(model, con[k])
        end
        unregister(model, :con)

    end
    
    return L
end



p = [ 13 10  3 16 12 11  1  9 19 13 ;     # profit 1
       1 10  3 13 12 19 16 13 11  9  ]    # profit 2
w  = [ 4, 4, 3, 5, 5, 3, 2, 3, 5, 4  ]    # weight
c  = 19                                   # capacity

rp = [40,40]
λ_ψ = fill([1.0238499551062132, 4.66018735258353],5)
solver = GLPK.Optimizer
#solver = HiGHS.Optimizer
#solver = Gurobi.Optimizer
#solver = CPLEX.Optimizer

start = time()
for _ in 1:20
    L(solver, p, w, c, rp, λ_ψ)
end
t_elapsed = time() - start
println(solver, " | time(s)= ", t_elapsed)
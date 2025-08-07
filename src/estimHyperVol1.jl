#using Distributions
#using JuMP, GLPK, Gurobi
#using SpecialFunctions


# =============================================================================
# Estimation of the hypervolume of an instance of a multi-objective 
# optimization problem without knowing its set of nondominated points.
#
# Example for a multi-objective 01 unidimensionnal knapsack problem.
#
# July 2025 
# version optimized - model modified iteratively-


# ------------------------------------------------------------
"""
    ψ(d)

Compute ψ according equation (17)
"""
function ψ(d::Int64)
    distribution = Normal(0,1)
    ϕ = rand(distribution,d)
    ϕ = abs.(ϕ)
    ϕnorm = sqrt(sum(ϕ[j]^2 for j in 1:d))
    return [ϕ[i]/ϕnorm for i in 1:d]
end

# ------------------------------------------------------------
"""
    λ(ψ)

Compute λ(ψ) according equation (18)
"""
function λ(ψ)    
    return [1.0/ψ[j] for j in 1:length(ψ)]
end

# ------------------------------------------------------------
"""
    L(p, w, c, rp, λ_ψ)

Compute a series of L(S,r_*,ψ) values according equations (18) and (20)
"""

function L(  solver::DataType, 
             p::Matrix{Int64}, w::Vector{Int64}, c::Int64,
             rp::Vector{Int64}, λ_ψ::Vector{Vector{Float64}}
          )

    L = (Float64)[]  # a series of L(S,r*,ψ) values

    # ------------------------------------------------------------------------------
    # Concerning the d-01UKP part of the model 
    #    a) setup the decision space X according (1), and
    #    b) declare the d objectives functions
    
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
        
        JuMP.optimize!(model)
        @assert is_solved_and_feasible(model) "Error: optimal solution not found"

        push!(L,objective_value(model))
        
        #println("zOpt: ", objective_value(model))
        #print("xOpt: ", value.(model[:x]))

        # delete the d constraints and unregister the name from the model
        for k=1:d
            delete(model, con[k])
        end
        unregister(model, :con)

    end
    
    return L
end

# ------------------------------------------------------------
"""
    H(p, w, c, rp=[0,0])

Compute H(S,r_*) according equation (14)
"""
                     
function H(solver, p,w,c, rp=[0,0], rndWeights=(100,200))
    numberOfWeights = rand(rndWeights[1]:rndWeights[2])  # the number of directions λ(ψ) 
    d = size(p,1) 
    λ_ψ = [λ(ψ(d)) for i=1:numberOfWeights]              # a list of weights 
    listL = L(solver, p, w, c, rp, λ_ψ)                  # Compute the optimal values of a series of Chebychev models over a random sample of λ(ψ)
    E = sum(listL.^d)/numberOfWeights                    # Estimate $E_{ψ ∈ Ψ^+}(L(S,r_*,ψ)^d)$
    H = 1/d * (2 * π^(d/2)) / ( gamma(d/2) * 2^d ) * E   # Compute $H(S,r_*)$ according (14)
    return H, numberOfWeights
end


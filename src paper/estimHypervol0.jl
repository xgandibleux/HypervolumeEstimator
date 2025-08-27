# =============================================================================
# Estimation of the hypervolume of an instance of a multi-objective 
# optimization problem without knowing its set of nondominated points.
#
# Example for a multi-objective 01 unidimensionnal knapsack problem.
#
# July 2025 
# version to be included in the paper - model rebuilt from scratch-


using Distributions
using JuMP, GLPK
using SpecialFunctions
Random.seed!(1234)


"""
    ψ(d)

Compute ψ according equation (17)
"""
function ψ(d)
    distribution = Normal(0,1)
    ϕ = rand(distribution,d)
    ϕ = abs.(ϕ)
    ϕnorm = sqrt(sum(ϕ[j]^2 for j in 1:d))
    return [ϕ[j]/ϕnorm for j in 1:d]
end


"""
    λ(ψ)

Compute λ(ψ) according equation (18)
"""
function λ(ψ)    
    return [1.0/ψ[j] for j in 1:length(ψ)]
end


"""
    L(p, w, c, rp, λ_ψ)

Compute a series of L(S,r_*,ψ) values according equations (18) and (20)
"""
function L(p, w, c, rp, λ_ψ)
    d,n = size(p)    
    m = Model(GLPK.Optimizer)  
    @variable(m, x[1:n], Bin)
    @constraint(m, sum(w[k]*x[k] for k=1:n) ≤ c)
    @expression(m, z[j=1:d], sum(p[j,k]*x[k] for k=1:n)) 
    @variable(m, α ≥ 0)        
    @objective(m, Max, α)
    @constraint(m, [j=1:d], α ≤ -λ_ψ[j]*(rp[j]-z[j]))
    optimize!(m)
    return objective_value(m)
end


"""
    H(p, w, c, rp=[0,0])

Compute H(S,r_*) according equation (14)
"""
function H(p, w, c, rp=[0,0])                      
    numberOfWeights = rand(5000:10000)   
    @show numberOfWeights
    d = size(p,1)  
    λ_ψ = [λ(ψ(d)) for i=1:numberOfWeights]
    listL = (Float64)[]
    for i in 1:length(λ_ψ)
        push!(listL, L(p, w, c, rp, λ_ψ[i]) ) 
    end
    E = sum(listL.^d)/numberOfWeights   
    H = 1/d * (2*π^(d/2)) / (gamma(d/2)*2^d) * E  
    return H
end


p = [ 13 10  3 16 12 11  1  9 19 13 ;     # profit 1
       1 10  3 13 12 19 16 13 11  9  ]    # profit 2
w  = [ 4, 4, 3, 5, 5, 3, 2, 3, 5, 4  ]    # weight
c  = 19                                   # capacity

rp = [0,0]
start = time()
H̃ = H(p, w, c, rp)
t_elapsed = time() - start

println("H estimated with rp=$rp : ", round(H̃, digits=2), " ($(round(t_elapsed, digits=2))s)")
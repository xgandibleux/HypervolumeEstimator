using Random
using JuMP, GLPK, HiGHS, Gurobi, CPLEX
import MultiObjectiveAlgorithms as MOA 
Random.seed!(1234)


function generate_MO01UKP(n = 10, o = 2, max_ci = 100, max_wi = 30)

    p = rand(1:max_ci,o,n) # c_i \in [1,max_ci]   # profits
    w = rand(1:max_wi,n) # w_i \in [1,max_wi]     # weight
    c = round(Int64, sum(w)/2)                    # capacity
                
    return p, w, c
end


function solve_MO01UKP(solver, p, w, c)

    o,n = size(p)
    mo01UKP = Model( )
    set_silent(mo01UKP)
    @variable(mo01UKP, x[1:n], Bin)
    @expression(mo01UKP, z[k=1:o], sum(p[k,j] * x[j] for j in 1:n))  
    @objective(mo01UKP, Max, [z[k] for k=1:o])
    @constraint(mo01UKP, sum(w[i] * x[i] for i in 1:n) ≤ c)

    set_optimizer(mo01UKP, () -> MOA.Optimizer(solver))
    #set_attribute(mo01UKP, MOA.Algorithm(), MOA.EpsilonConstraint())
    #set_attribute(mo01UKP, MOA.Algorithm(), MOA.KirlikSayin())
    set_attribute(mo01UKP, MOA.Algorithm(), MOA.TambyVanderpooten())
    #set_attribute(mo01UKP, MOA.Algorithm(), MOA.DominguezRios())
    optimize!(mo01UKP)

    @assert is_solved_and_feasible(mo01UKP) "Error: optimal solution not found"
    S = (Vector{Int64})[]
    for i in 1:result_count(mo01UKP)
        push!(S, round.(Int, objective_value(mo01UKP; result = i)) )
    end

    return S, result_count(mo01UKP)

end


solver = GLPK.Optimizer
#solver = HiGHS.Optimizer
#solver = Gurobi.Optimizer
#solver = CPLEX.Optimizer
n = 30    # number of variables
o = 2     # number of objectives


println("  number of variables  : ", n)
println("  number of objectives : ", o)
println("  solver MIP invoked   : ", solver)

println("\nGenerate an mo01UKP instance...")
p, w, c = generate_MO01UKP(n,o)

println("\nCompute S, the set of nondominated points...")
start = time()
S, cardS = solve_MO01UKP(solver, p, w, c)
t_elapsedS = round(time() - start, digits=2)
println("  |S|  = ",cardS, " ($t_elapsedS s)")

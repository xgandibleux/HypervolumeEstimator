using Random
using JuMP, GLPK#, HiGHS, Gurobi, CPLEX
import MultiObjectiveAlgorithms as MOA 
using Plots
using LaTeXStrings
Random.seed!(1234)


function generate_MO01UKP(n = 10, o = 2, max_ci = 100, max_wi = 30)

    p = rand(1:max_ci,o,n) # c_i \in [1,max_ci]   # profits
    w = rand(1:max_wi,n) # w_i \in [1,max_wi]     # weight
    c = round(Int64, sum(w)/2)                    # capacity
                
    return p, w, c
end

function set_MO01UKP()
    p = [
        13 10  3 16 12 11  1  9 19 13
        1 10  3 13 12 19 16 13 11  9
    ]
    w = [4, 4, 3, 5, 5, 3, 2, 3, 5, 4]
    c = 19
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
#p, w, c = generate_MO01UKP(n,o)
p, w, c = set_MO01UKP()

println("\nCompute S, the set of nondominated points...")
start = time()
S, cardS = solve_MO01UKP(solver, p, w, c)
t_elapsedS = round(time() - start, digits=2)
println("  |S|  = ",cardS, " ($t_elapsedS s)")


z1nd = [s[1] for s in S]
z2nd = [s[2] for s in S]

scatter(z1nd, z2nd, 
        xlim=(35,75), ylim=(35, 75),
        color="blue", 
        title = "Objective space", 
        label = L"Y_N",
        #legend = :bottomleft,
        aspect_ratio=:equal)
xlabel!("objective 1")
ylabel!("objective 2")
savefig("objSpace.png")


#scatter!(z1nd, z2nd, markershape = :x , color="blue", markersize = 10, label = "NSGA-II: "*L"Z_N")
# plot the set of nondominated points found by MOA with the TambyVanderpooten algorithm
#z1exact = [305, 324, 299, 292, 325, 273, 298]  
#z2exact = [269, 265, 274, 295, 240, 299, 290] 
#scatter!(z1exact, z2exact, markershape = :cross , color="red", markersize = 8, label = "TambyVanderpooten: "*L"Z_N", markerstrokewidth = 2)

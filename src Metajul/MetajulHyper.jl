using MetaJul
using Random
using LaTeXStrings

#Random.seed!(42)
#rows, cols = 2, 100
#profits = rand(5:20, rows, cols)
#weights = rand(1:20, 1, cols)
#capacities = [150; ]

rows, cols = 2, 10
profits = [63 31 12 32 6 54 38 55 83 47; 
           71 30 34 5 39 53 9 40 62 9 ]
weights = [12 16 14 23 24 8 14 20 2 30; ] 
capacities = [82; ]

biObjectiveKnapsack = multiObjectiveKnapsack(profits, weights, capacities);
println("Number of variables  : ", numberOfVariables(biObjectiveKnapsack))
println("Number of objectives : ", numberOfObjectives(biObjectiveKnapsack))
println("Nnmber of constraints: ", numberOfConstraints(biObjectiveKnapsack))

solver::NSGAII = NSGAII(
               biObjectiveKnapsack, 
               populationSize = 100, 
               termination = TerminationByEvaluations(900),
               mutation = BitFlipMutation(probability = 1.0 / biObjectiveKnapsack.numberOfBits),
               crossover = SinglePointCrossover(probability = 1.0)
               )
solver.dominanceComparator = ConstraintsAndDominanceComparator();

optimize!(solver) ;


solutionsAreFeasible = all(solution -> isFeasible(solution), foundSolutions(solver))
z1 = [-1 * solution.objectives[1] for solution in foundSolutions(solver)];
z2 = [-1 * solution.objectives[2] for solution in foundSolutions(solver)];


using Plots

scatter(z1, z2, 
        xlim=(200,330), ylim=(200, 330),
        color="blue", 
        title = "Objective space", 
        label = "NSGA-II: final population",
        legend = :bottomleft,
        aspect_ratio=:equal)
xlabel!("objective 1")
ylabel!("objective 2")


# plot the nondominated points within the final population returned by NSGA-II
archive = NonDominatedArchive(BinarySolution)
for solution in foundSolutions(solver)
        @show solution
        add!(archive, solution)
end
@show archive
for solution in getSolutions(archive)
        @show solution
end
z1nd = [-1 * solution.objectives[1] for solution in getSolutions(archive)];
z2nd = [-1 * solution.objectives[2] for solution in getSolutions(archive)];

scatter!(z1nd, z2nd, markershape = :x , color="blue", markersize = 10, label = "NSGA-II: "*L"Z_N")


# plot the set of nondominated points found by MOA with the TambyVanderpooten algorithm
z1exact = [305, 324, 299, 292, 325, 273, 298]  
z2exact = [269, 265, 274, 295, 240, 299, 290] 
scatter!(z1exact, z2exact, markershape = :cross , color="red", markersize = 8, label = "TambyVanderpooten: "*L"Z_N", markerstrokewidth = 2)
#savefig("metajul.png")

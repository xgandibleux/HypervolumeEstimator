using MetaJul
using Random

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
        title = "Pareto front approximation", 
        label = "NSGA-II")
xlabel!("Profit 1")
ylabel!("Profit 2")


z1exact = [305, 324, 299, 292, 325, 273, 298]  
z2exact = [269, 265, 274, 295, 240, 299, 290] 
scatter!(z1exact, z2exact, markershape = :cross , color="red", markersize = 8, label = "TambyVanderpooten")
savefig("metajul.png")
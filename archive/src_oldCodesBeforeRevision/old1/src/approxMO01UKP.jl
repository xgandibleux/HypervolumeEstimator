
"""
    approxMO01UKP(p, w, c)

Using MetaJul for computing an approximation of Y_N with NSGA-II
"""
function approxMO01UKP(p, w, c)

        rows, cols = size(p)

        # prepare the inputs according the expected formats ---------
        profits = deepcopy(p)
        weights = reshape(w,(1,:))
        capacities = [c; ]

        biObjectiveKnapsack = multiObjectiveKnapsack(profits, weights, capacities)
        solver::NSGAII = NSGAII(
                biObjectiveKnapsack, 
                populationSize = 2 * cols * rows, #100, 
                termination = TerminationByEvaluations(50000),
                mutation = BitFlipMutation(probability = 1.0 / biObjectiveKnapsack.numberOfBits),
                crossover = SinglePointCrossover(probability = 1.0)
                )
        solver.dominanceComparator = ConstraintsAndDominanceComparator();

        observer = EvaluationObserver(1000)
        register!(observable(solver), observer)

        MetaJul.optimize!(solver)
        solutionsAreFeasible = all(solution -> isFeasible(solution), foundSolutions(solver))


        # extract the nondominated points within the final population returned by NSGA-II
        archive = NonDominatedArchive(BinarySolution)
        for solution in foundSolutions(solver)
                add!(archive, solution)
        end


        # return the outputs according the expected formats ---------
        znd = Array{Int}(undef,length(archive), rows)
        for k in 1:rows
                znd[:,k] = [-1 * solution.objectives[k] for solution in getSolutions(archive)]
        end
        #z2nd = [-1 * solution.objectives[2] for solution in getSolutions(archive)];
        #@show znd

        S = (Vector{Int64})[]
        for i in 1:length(archive)
                println("$i : z = ", [znd[i,j] for j in 1:rows])
                push!(S, [znd[i,j] for j in 1:rows] )
        end
        #@show S

        return S, length(archive)
end
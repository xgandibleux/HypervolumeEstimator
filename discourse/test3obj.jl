using JuMP
import Gurobi
import MultiObjectiveAlgorithms as MOA 

function main()
    p = [
        33 90 96 75 1 69 100 50 63 61 59 95 58 10 77 30 86 89 82 51 38 33 73 54 91 89 95 82 48 67 
        55 36 80 58 20 96 75 57 24 68 37 58 8 85 27 25 71 53 47 72 57 64 1 8 12 68 3 80 20 90 
        22 40 50 73 44 65 12 26 13 77 14 68 71 35 54 98 45 95 98 19 18 38 14 51 37 48 35 97 95 36 
    ]
    w = [22, 13, 10, 25, 4, 15, 17, 15, 15, 28, 14, 13, 2, 23, 6, 22, 18, 6, 23, 21, 7, 7, 14, 4, 3, 27, 10, 5, 9, 10]
    model = Model(() -> MOA.Optimizer(Gurobi.Optimizer))
    set_attribute(model, MOA.Algorithm(), MOA.TambyVanderpooten())
    set_silent(model)
    set_time_limit_sec(model, 60.0)
    @variable(model, x[1:length(w)], Bin)
    @objective(model, Max, p * x)
    @constraint(model, w' * x <= round(Int, sum(w) / 2))
    optimize!(model)
    assert_is_solved_and_feasible(model)
    return map(1:result_count(model)) do result
        return round.(Int, objective_value(model; result))
    end
end

@time S = main()
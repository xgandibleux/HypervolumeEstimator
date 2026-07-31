# =============================================================================
# Unbiased hypervolume estimator for the bi-objective 0-1 knapsack problem.
# Parallelised version — self-contained illustration.
#
# Usage: julia --threads 8 hv_ukp_illustration_threaded_vf.jl
# =============================================================================

using Random, LinearAlgebra
using JuMP, Gurobi
using SpecialFunctions, HypothesisTests, Statistics
using Base.Threads

# ψ(ukp, rng) : random direction on the positive unit quarter-sphere
ψ(ukp, rng) = (ϕ = abs.(randn(rng, ukp.d)); ϕ / norm(ϕ))

# λ(v) : amplification vector
λ(v) = 1.0 ./ v

# =============================================================================
function L_threaded(ukp, rp, N)

    listL  = Vector{Float64}(undef, N)
    chunks = collect(Iterators.partition(1:N, ceil(Int, N / Threads.nthreads())))
    rngs   = [MersenneTwister(1234 + t) for t in 1:Threads.maxthreadid()]

    Threads.@threads for chunk in chunks

        rng = rngs[Threads.threadid()]

        # one model per thread, built once
        m = Model(Gurobi.Optimizer)
        set_silent(m)
        set_attribute(m, "Threads", 1)    # avoid oversubscription with T concurrent models
        set_attribute(m, "MIPGap",  0.0)  # exact optimality for each L evaluation

        @variable(m, x[1:ukp.n], Bin)
        @constraint(m, sum(ukp.w[i] * x[i] for i in 1:ukp.n) ≤ ukp.c)
        @expression(m, z[k=1:ukp.d], sum(ukp.p[k,j] * x[j] for j in 1:ukp.n))
        @variable(m, α ≥ 0)
        @objective(m, Max, α)

        for i in chunk
            λ_ψ = λ(ψ(ukp, rng))
            @constraint(m, con[k=1:ukp.d], α ≤ -λ_ψ[k] * (rp[k] - z[k]))
            optimize!(m)
            @assert is_solved_and_feasible(m) "STOP: optimal solution not found" 
            listL[i] = objective_value(m)
            for k in 1:ukp.d; delete(m, con[k]); end
            unregister(m, :con)
        end

    end

    return listL
end

# =============================================================================
function H(ukp, rp, N, Hexact)
    list_L  = L_threaded(ukp, rp, N)
    coeff   = 1/ukp.d * (2 * π^(ukp.d/2)) / (gamma(ukp.d/2) * 2^ukp.d)
    H_est   = coeff * sum(l^ukp.d for l in list_L) / N
    lw_norm = coeff / Hexact .* [l^ukp.d for l in list_L]
    CIlow, CIhigh = confint(OneSampleTTest(lw_norm), level=0.95, tail=:both)
    return H_est, (mean(lw_norm), CIlow, CIhigh)
end

# =============================================================================
function main()

    ukp = ( p = [ 13 10  3 16 12 11  1  9 19 13 ;   # profits objective 1
                   1 10  3 13 12 19 16 13 11  9  ],  # profits objective 2
            w = [  4,  4,  3,  5,  5,  3,  2,  3,  5,  4 ],  # weights
            c = 19,                                            # capacity
            d = 2,                                             # number of objectives
            n = 10 )                                           # number of variables

    rp     = [0, 0]    # reference point
    Hexact = 4637.0    # exact hypervolume

    println("Instance : $(ukp.n) items, $(ukp.d) objectives, capacity = $(ukp.c)")
    println("Reference point : rp = $rp")
    println("H exact = $Hexact")
    println("Threads : ", Threads.nthreads())

    Random.seed!(1234)
    N = 500
    println("\nNumber of directions N = $N")

    start   = time()
    H_est, (H_norm, CIlow, CIhigh) = H(ukp, rp, N, Hexact)
    elapsed = round(time() - start, digits=2)

    println("\nH estimated            = $(round(H_est,  digits=2))")
    println("H estimated normalized = $(round(H_norm, digits=5))")
    println("CI 95%                 = [$(round(CIlow, digits=5)), $(round(CIhigh, digits=5))]")
    println("Relative error         = $(round(abs(1.0 - H_norm), digits=5))")
    println("Elapsed                = $(elapsed) s")
end

main()

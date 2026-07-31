# =============================================================================
# Hypervolume estimation of an unknown TPF — improved and distributed version
# (Section 4.3, revised)
#
# Run:  julia --threads 8 hv_ukp_parallel_rev2.jl
#
# Requires the functions ψ, λ of the sequential version (Section 4.2, revised).
# =============================================================================

using Random, LinearAlgebra
using JuMP, Gurobi
using SpecialFunctions, HypothesisTests, Statistics

# --- Function for computing ψ according (19) — as in Section 4.2 (revised) --
function ψ(ukp, rng::AbstractRNG = Random.default_rng())
    ϕ = abs.(randn(rng, ukp.d))
    ϕ = max.(ϕ, 1e-12)              # zero-coordinate guard (Section 3.2)
    return ϕ / norm(ϕ)
end

# --- Function for computing λ(ψ) according (24) — as in Section 4.2 ---------
λ(v) = 1.0 ./ v

# --- Distributed evaluation of the N distances L_i ---------------------------
const CHUNKSIZE = 50   # fixed, independent of the number of threads

function L_threaded(ukp, rp, N)
    listL = Vector{Float64}(undef, N)
    chunks = collect(
                Iterators.partition( 1:N,
                    CHUNKSIZE
                )
             )

    Threads.@threads for chunk_id in eachindex(chunks)
        rng = MersenneTwister(1234 + chunk_id)          # one RNG per chunk

        m = Model(Gurobi.Optimizer)
        set_silent(m)
        set_attribute(m, "Threads", 1)
        set_attribute(m, "MIPGap", 0.0)

        @variable(m, x[1:ukp.n], Bin)
        @constraint(m, sum(ukp.w[i]*x[i] for i in 1:ukp.n) ≤ ukp.c)
        @expression(m, z[k=1:ukp.d], sum(ukp.p[k,j]*x[j] for j in 1:ukp.n))
        @variable(m, α ≥ 0)
        @objective(m, Max, α)

        for i in chunks[chunk_id]
            λ_ψ = λ(ψ(ukp, rng))
            @constraint(m, con[k=1:ukp.d], α ≤ -λ_ψ[k]*(rp[k]-z[k]))
            optimize!(m)
            @assert is_solved_and_feasible(m) "STOP: opt. sol. not found"
            listL[i] = objective_value(m)
            for k in 1:ukp.d; delete(m, con[k]); end
            unregister(m, :con)
        end
    end
    return listL
end

# --- Function for computing H according (18) — as in the paper --------------
function H(ukp, rp, N, Hexact)
    list_L = L_threaded(ukp, rp, N)
    coeff = 1/ukp.d * (2 * π^(ukp.d/2)) / (gamma(ukp.d/2) * 2^ukp.d)
    H_est = coeff * sum(l^ukp.d for l in list_L) / N
    lw_norm = coeff / Hexact .* [l^ukp.d for l in list_L]
    CIlow, CIhigh = confint(OneSampleTTest(lw_norm), level=0.95, tail=:both)
    return H_est, (mean(lw_norm), CIlow, CIhigh)
end

# --- Entry point: didactic instance (Section 4.1) ----------------------------
function main()
    ukp = (
        p = [ 13 10 3 16 12 11  1  9 19 13 ;   # profits objective 1
               1 10 3 13 12 19 16 13 11  9 ],  # profits objective 2
        w = [ 4, 4, 3, 5, 5, 3, 2, 3, 5, 4 ],  # weights
        c = 19,                                 # capacity
        d = 2,                                  # number of objectives
        n = 10                                  # number of variables
    )

    rp = [40, 40]        # reference point
    Hexact = 717.0       # exact hypervolume

    N = 500              # number of directions

    start = time()
    H_est, (H_norm, CIlow, CIhigh) = H(ukp, rp, N, Hexact)
    elapsed = round(time() - start, digits=2)

    println("threads                = ", Threads.nthreads())
    println("H estimated            = ", round(H_est, digits=2))
    println("H estimated normalized = ", round(H_norm, digits=5))
    println("CI 95%                 = [", round(CIlow, digits=5),
            " ; ", round(CIhigh, digits=5), "]")
    println("relative error         = ", round(abs(1 - H_norm), digits=5))
    println("elapsed (s)            = ", elapsed)
end

main()

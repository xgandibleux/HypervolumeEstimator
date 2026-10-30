# =============================================================================
# Common functions shared by all hypervolume estimators (UKP, UFLP, ...).
# -revT: zero-coordinate guard added in ψ! (Section 3.2 of the paper);
#        CHUNKSIZE and set_exact_and_singlethread! shared by the parallel
#        estimators of all problems.


# ------------------------------------------------------------
"""
    ψ!(buf, rng)

Compute ψ in-place according equation (17).
Writes into buf — no allocation.
Any coordinate below 1e-12 (in particular an exact zero, which would
yield an infinite weight) is replaced by this small positive value.
"""
function ψ!(buf::Vector{Float64}, rng::AbstractRNG)
    d = length(buf)
    @inbounds for j in 1:d
        buf[j] = max(abs(randn(rng)), 1e-12)   # zero-coordinate guard
    end
    nrm = sqrt(sum(abs2, buf))
    @inbounds buf ./= nrm
    return buf
end


# ------------------------------------------------------------
"""
    λ!(lam, psi)

Compute λ(ψ) in-place according equation (18).
Writes into lam — no allocation.
"""
function λ!(lam::Vector{Float64}, psi::Vector{Float64})
    @inbounds for j in 1:length(psi)
        lam[j] = 1.0 / psi[j]
    end
    return lam
end


# ------------------------------------------------------------
# chunk size used by all parallel estimators: fixed, independent of the
# number of threads, so that the sampled directions (hence the estimates)
# are identical for any value of --threads
const CHUNKSIZE = 50


# ------------------------------------------------------------
"""
    set_exact_and_singlethread!(model, solver)

Configure the MIP solver of `model` for the parallel estimation:
exactly one internal thread (parallelism is handled by Threads.@threads;
without this, multi-instance solving oversubscribes the cores) and a
zero MIP gap tolerance (exact optimality of each L_i, avoiding a
systematic downward bias of the estimator).
"""
function set_exact_and_singlethread!(model, solver::DataType)
    if occursin("Gurobi", string(solver))
        set_attribute(model, "Threads", 1)
        set_attribute(model, "MIPGap", 0.0)
    elseif occursin("HiGHS", string(solver))
        set_attribute(model, "threads", 1)
        set_attribute(model, "mip_rel_gap", 0.0)
    end
    return model
end

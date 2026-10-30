# =============================================================================
# Common functions shared by all hypervolume estimators (UKP, UFLP, ...).


# ------------------------------------------------------------
"""
    ψ!(buf, rng)

Compute ψ in-place according equation (17).
Writes into buf — no allocation.
"""
function ψ!(buf::Vector{Float64}, rng::MersenneTwister)
    d = length(buf)
    @inbounds for j in 1:d
        buf[j] = abs(randn(rng))
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

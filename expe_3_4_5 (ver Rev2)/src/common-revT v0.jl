# =============================================================================
# Common functions shared by all hypervolume estimators (UKP, UFLP, ...).
# -revT: zero-coordinate guard added in ψ! (Section 3.2 of the paper).


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

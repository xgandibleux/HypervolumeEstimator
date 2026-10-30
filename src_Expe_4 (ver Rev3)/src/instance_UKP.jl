# =============================================================================
# Instance structure and generators for the multi-objective 0-1 knapsack
# problem (MO-01UKP).

mutable struct Instance_UKP
    fname :: String
    n     :: Int64          # number of items
    o     :: Int64          # number of objectives
    p     :: Matrix{Int64}  # profits  (o × n)
    w     :: Vector{Int64}  # weights  (n)
    c     :: Int64          # capacity
end


# ------------------------------------------------------------
"""
    generate_MO01UKP(n=10, o=2, max_ci=100, max_wi=30)

Generate randomly an instance for the MO-01UKP.
Profits drawn from [1, max_ci], weights from [1, max_wi], capacity = sum(w)/2.
"""
function generate_MO01UKP(n=10, o=2, max_ci=100, max_wi=30)

    p = rand(1:max_ci, o, n)
    w = rand(1:max_wi, n)
    c = round(Int64, sum(w) / 2)

    return Instance_UKP("generated", n, o, p, w, c)
end


# ------------------------------------------------------------
"""
    reference_point_LB(inst::Instance_UKP)

Compute an improved reference point for the hypervolume estimator using
the LB bound of Glover (1965).

nMin is a lower bound on the number of items of any efficient solution (it assumes that
every item has a strictly positive profit on at least one objective): sort items by
decreasing weight and select the heaviest items, STOPPING AT THE FIRST ITEM THAT DOES NOT FIT,
i.e. nMin = max{k : w_(1) + ... + w_(k) <= c}.
For each objective k, the nMin smallest profit coefficients are summed
independently to form the k-th component of the reference point.

Returns rp::Vector{Int64} of length inst.o.
"""
function reference_point_LB(inst::Instance_UKP)

    # sort items by decreasing weight (heaviest first)
    idx = sortperm(inst.w, rev=true)

    # select the heaviest items until the first one that does not fit : gives nMin items
    capacity_used = 0
    selected = Int64[]
    for i in idx
        capacity_used + inst.w[i] <= inst.c || break   # stop at the FIRST item that does not fit
        push!(selected, i)
        capacity_used += inst.w[i]
    end

    # nMin : number of items in the LB selection (no subtraction)
    nMin = length(selected)

    # for each objective k, sort coefficients in increasing order and
    # sum the nMin smallest values
    rp = [sum(sort(inst.p[k, :])[1:nMin]) for k in 1:inst.o]

    return rp
end

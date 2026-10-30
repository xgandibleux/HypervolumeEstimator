# =============================================================================
# Instance structure and generator for the simple continuous multi-objective
# linear problem used in Experiment 6 (weighted simplex):
#
#   max   z_k = a_k * x_k          k = 1..d
#   s.t.  sum_i b_i * x_i <= c
#         x_i >= 0 (continuous)
#
# The set of attainable points, completed downwards in the nonnegative orthant,
# is the simplex { z >= 0 : sum_k (b_k / a_k) z_k <= c }, whose vertices lie on
# the axes at z_k = a_k * c / b_k. With the origin as reference point:
#
#   H = prod_k (a_k * c / b_k) / d!
#
# so the exact hypervolume is known in closed form (no call to HV).

mutable struct Instance_LP
    fname :: String
    d     :: Int64            # number of objectives (= number of variables)
    a     :: Vector{Int64}    # objective scaling  (d)
    b     :: Vector{Int64}    # constraint coefficients (d)
    c     :: Int64            # right-hand side
end


# ------------------------------------------------------------
"""
    generate_LP(d=3; random=false, maxval=100)

Generate an instance of the weighted simplex problem.
  - random=false : a_k = b_k = c = 1 (H = 1/d!)
  - random=true  : a_k, b_k, c drawn uniformly in [1, maxval] (integers)
"""
function generate_LP(d::Int=3; random::Bool=false, maxval::Int=100)

    if random
        a = rand(1:maxval, d)
        b = rand(1:maxval, d)
        c = rand(1:maxval)
        return Instance_LP("generated", d, a, b, c)
    else
        return Instance_LP("unit", d, ones(Int64, d), ones(Int64, d), 1)
    end
end


# ------------------------------------------------------------
"""
    exact_hypervolume(inst::Instance_LP)

Closed-form hypervolume of the dominated region with respect to the origin:
H = prod_k (a_k * c / b_k) / d!
"""
function exact_hypervolume(inst::Instance_LP)
    vertices = [inst.a[k] * inst.c / inst.b[k] for k in 1:inst.d]
    return prod(vertices) / factorial(inst.d)
end


# ------------------------------------------------------------
"""
    L_analytic(inst::Instance_LP, psi)

Closed-form value of L(S, 0, ψ) for the weighted simplex, with λ = 1/ψ:
    L = c / sum_k (b_k * ψ_k / a_k)
Used to check the LP solutions.
"""
function L_analytic(inst::Instance_LP, psi::Vector{Float64})
    return inst.c / sum(inst.b[k] * psi[k] / inst.a[k] for k in 1:inst.d)
end

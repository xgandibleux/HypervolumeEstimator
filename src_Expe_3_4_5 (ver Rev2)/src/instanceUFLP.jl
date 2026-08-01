# =============================================================================
# Instance structure and I/O for the multi-objective Uncapacitated Facility
# Location Problem (UFLP), variant 2: o independent cost sets on the
# same constraint structure.

mutable struct Instance_UFLP
    fname :: String
    nI    :: Int64                  # number of users
    nJ    :: Int64                  # number of services
    o     :: Int64                  # number of objectives
    c     :: Vector{Matrix{Int64}}  # c[k] : assignment costs (nI × nJ), objective k
    r     :: Vector{Vector{Int64}}  # r[k] : fixed running costs (nJ),   objective k
end


# ------------------------------------------------------------
"""
    generate_UFLP(nI, nJ, o=2; max_c=100, max_r=200)

Generate a random multi-objective UFLP instance with o objectives.
  nI    : number of users
  nJ    : number of services
  o     : number of objectives (default 2)
Assignment costs drawn from [1, max_c], fixed costs from [1, max_r].
"""
function generate_UFLP(nI::Int, nJ::Int, o::Int=2;
                       max_c::Int=100, max_r::Int=200)
    c = [rand(1:max_c, nI, nJ) for _ in 1:o]
    r = [rand(1:max_r, nJ)     for _ in 1:o]
    return Instance_UFLP("generated", nI, nJ, o, c, r)
end


# ------------------------------------------------------------
"""
    load_UFLP(fdirectory, fname)

Load a multi-objective UFLP instance from a file.
File format:
  line 1 : nI  (number of users)
  line 2 : nJ  (number of services)
  line 3 : o   (number of objectives)
  then for each objective k = 1..o:
    blank line
    nI lines : assignment costs objective k, one row per user (nJ values)
    blank line
    1 line   : fixed costs objective k  (nJ values)
"""
function load_UFLP(fdirectory::String, fname::String)
    f = open(fdirectory * "/" * fname)

    nI = parse(Int, readline(f))
    @assert nI ≤ typemax(UInt16) "STOP: Maximum 65535 users"
    nJ = parse(Int, readline(f))
    o  = parse(Int, readline(f))

    c = Vector{Matrix{Int64}}(undef, o)
    r = Vector{Vector{Int64}}(undef, o)

    for k in 1:o
        useless  = readline(f)
        ck = Matrix{Int64}(undef, nI, nJ)
        for i in 1:nI
            ck[i, :] = parse.(Int64, split(readline(f)))
        end
        c[k] = ck
        useless  = readline(f)
        r[k] = parse.(Int64, split(readline(f)))
    end

    close(f)
    return Instance_UFLP(fname, nI, nJ, o, c, r)
end


# ------------------------------------------------------------
"""
    reference_point(inst::Instance_UFLP)

Compute the naive upper bound reference point for the UFLP estimator.
For each objective k:
  UB_k = sum(r[k]) + sum_i max_j(c[k][i,j])
Returns rp = [-UB_1, ..., -UB_o] (negated for the maximisation-based estimator).
Computed per instance.
"""
function reference_point(inst::Instance_UFLP)
    rp = Vector{Int64}(undef, inst.o)
    for k in 1:inst.o
        UB_k = sum(inst.r[k]) +
               sum(maximum(inst.c[k][i, :]) for i in 1:inst.nI)
        rp[k] = -UB_k
    end
    return rp
end

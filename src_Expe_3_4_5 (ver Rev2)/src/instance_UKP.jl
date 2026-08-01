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
    didactic_MO01UKP()

Setup a didactic instance of MO-01UKP (10 items, 2 objectives).
"""
function didactic_MO01UKP()

    p = [ 13 10  3 16 12 11  1  9 19 13 ;     # profit 1
           1 10  3 13 12 19 16 13 11  9  ]    # profit 2
    w  = [ 4, 4, 3, 5, 5, 3, 2, 3, 5, 4  ]   # weight
    c  = 19                                   # capacity

    return Instance_UKP("didactic", 10, 2, p, w, c)
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
    TamVan_MO01UKP()

Setup the instance MOKP_p-6_n-30_1.dat of MO-01UKP (30 items, 6 objectives).
"""
function TamVan_MO01UKP()

    p = [ 42  74  52  57  41  83  89  74  87  49  3  9  57  18  26  8  26  5  76  26  19  87  81  28  89  10  58  71  49  21;
          34  84  51  77  99  70  48  72  60  85  45  36  9  51  70  51  58  12  47  86  21  39  20  96  97  40  16  74  34  3;
          99  98  8  95  84  96  72  60  64  3  63  62  13  26  96  67  99  26  4  35  3  11  81  77  26  3  57  38  81  72;
          85  47  66  66  76  18  44  91  94  75  20  96  28  61  39  89  48  97  43  27  3  18  29  18  18  21  51  60  61  51;
          12  8  25  83  16  77  99  25  25  25  66  69  60  31  2  26  44  17  42  18  47  94  71  84  12  58  15  74  38  72;
          74  98  68  17  20  39  42  95  10  88  59  23  27  73  55  2  40  23  98  58  45  19  99  89  89  82  69  84  49  62]
    w  = [9, 45, 74, 75, 86, 5, 62, 89, 84, 5, 75, 41, 77, 68, 7, 75, 84, 97, 86, 79, 48, 44, 15, 22, 11, 44, 77, 22, 34, 27]
    c  = 783

    return Instance_UKP("TamVan", 30, 6, p, w, c)
end


# ------------------------------------------------------------
"""
    reference_point_LB0(inst::Instance_UKP)

Compute an improved reference point for the hypervolume estimator using
the LB bound of Glover (1965) as described in Gandibleux & Freville (2000).

LB is the minimum number of items in any efficient solution: sort items by
decreasing weight, select items greedily until the knapsack is full.
The objective values of this selection, minus 1 on each coordinate, form a
reference point strictly dominated by all nondominated points.

Returns rp::Vector{Int64} of length inst.o.
"""
function reference_point_LB0(inst::Instance_UKP)

    # sort items by decreasing weight (heaviest first)
    idx = sortperm(inst.w, rev=true)

    # select items greedily until capacity is exceeded : gives LB items
    capacity_used = 0
    selected = Int64[]
    for i in idx
        if capacity_used + inst.w[i] <= inst.c
            push!(selected, i)
            capacity_used += inst.w[i]
        end
    end

    # evaluate the o objectives on the selected items, subtract 1 to ensure
    # strict domination (eliminates any weak dominance case)
    rp = [sum(inst.p[k, i] for i in selected) - 1 for k in 1:inst.o]

    return rp
end


"""
    reference_point_LB(inst::Instance_UKP)

Compute an improved reference point for the hypervolume estimator using
the LB bound of Glover (1965).

LB is the minimum number of items in any efficient solution: sort items by
decreasing weight, select items greedily until the knapsack is full.
nMin = LB is then used as a lower bound on the number of selected items.
For each objective k, the nMin smallest profit coefficients are summed
independently to form the k-th component of the reference point.

Returns rp::Vector{Int64} of length inst.o.
"""
function reference_point_LB(inst::Instance_UKP)

    # sort items by decreasing weight (heaviest first)
    idx = sortperm(inst.w, rev=true)

    # select items greedily until capacity is exceeded : gives LB items
    capacity_used = 0
    selected = Int64[]
    for i in idx
        if capacity_used + inst.w[i] <= inst.c
            push!(selected, i)
            capacity_used += inst.w[i]
        end
    end

    # nMin : number of items in the LB selection (no subtraction)
    nMin = length(selected)

    # for each objective k, sort coefficients in increasing order and
    # sum the nMin smallest values
    rp = [sum(sort(inst.p[k, :])[1:nMin]) for k in 1:inst.o]

    return rp
end
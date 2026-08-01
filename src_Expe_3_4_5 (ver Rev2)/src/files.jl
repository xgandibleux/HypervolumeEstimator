# ------------------------------------------------------------
# Path to the shared library built from hv-1.3-src via build_lib.sh/.bat
# Platform-specific extension: .dylib (macOS), .so (Linux), .dll (Windows)
const LIBFPLI_HV = joinpath(@__DIR__,
    Sys.iswindows() ? "fpli_hv.dll"      :
    Sys.isapple()   ? "libfpli_hv.dylib" :
                      "libfpli_hv.so"    )

"""
    compute_Hmeasure(S, o, rp=zeros(Int,o))

Compute the exact hypervolume for maximisation problems (e.g. UKP).
Points are negated before calling fpli_hv (which minimises internally).
rp is the reference point in the original maximisation space (default: origin).
rp must be strictly dominated by all points in S.

For an improved reference point, pass rp = reference_point_LB(p, w, c)
which uses the LB-selection of the heaviest items (Glover 1965,
Gandibleux & Freville 2000).
"""
function compute_Hmeasure(S::Vector{Vector{Int64}}, o::Int,
                          rp::Vector{Int64}=zeros(Int64, o))
    n    = length(S)
    data = Vector{Float64}(undef, o * n)
    @inbounds for i in 1:n
        for k in 1:o
            data[o * (i-1) + k] = Float64(-S[i][k])   # negate for fpli_hv
        end
    end
    ref = Float64.(-rp)    # negate rp to match the negated points space
    hv  = @ccall LIBFPLI_HV.fpli_hv(
              data::Ptr{Float64},
              o::Cint,
              n::Cint,
              ref::Ptr{Float64}
          )::Cdouble
    return hv
end


"""
    compute_Hmeasure_min(S, o, ref)

Compute the exact hypervolume for minimisation problems (e.g. UFLP).
Points are passed directly without negation (fpli_hv natively minimises).
ref must strictly dominate all points: ref[k] > S[i][k] for all i, k.

For UFLP, pass ref = Float64.(-rp) where rp = reference_point(inst)
(negated UB bounds) — this matches the reference point used by the estimator.
"""
function compute_Hmeasure_min(S::Vector{Vector{Int64}}, o::Int,
                               ref::Vector{Float64})
    n    = length(S)
    data = Vector{Float64}(undef, o * n)
    @inbounds for i in 1:n
        for k in 1:o
            data[o * (i-1) + k] = Float64(S[i][k])    # no negation
        end
    end
    hv  = @ccall LIBFPLI_HV.fpli_hv(
              data::Ptr{Float64},
              o::Cint,
              n::Cint,
              ref::Ptr{Float64}
          )::Cdouble
    return hv
end


# ------------------------------------------------------------
# save the data describing an instance generated
function save_instance(fname::String, p::Matrix{Int64}, w::Vector{Int64}, c::Int64)
    d,n = size(p)       # number of objectives and number of variables
    open(fname, "w") do io
        write(io, string(n, " ", d, "\n")) # number of variables

        # Saving the vector of profits, 1 objective per line
        for k in 1:d
            for j in 1:n
                write(io, string(p[k,j], " "))
            end
            write(io, "\n")
        end

        # Saving the vector of weights 
        for val in w
            write(io, string(val, " "))
        end
        write(io, "\n")

        write(io, string(c, "\n")) # RHS
    end

    return nothing
end

# ------------------------------------------------------------
# save the data describing an instance generated
function save_nondominatedpoints(fname::String, S::Vector{Vector{Int64}})
    n = length(S)       # number of points
    open(fname, "w") do io
        write(io, string(n, "\n")) # number of points

        # Saving the points, 1 point per line
        for p in 1:n
            for k in 1:length(S[p])
                write(io, string(S[p][k], " "))
            end
            write(io, "\n")
        end
    end

    return nothing
end

# ------------------------------------------------------------
# save the data describing an instance generated
function save_nondominatedpoints(fname::String, S::Set{Vector{Int64}})
    n = length(S)       # number of points
    open(fname, "w") do io
        write(io, string(n, "\n")) # number of points

        # Saving the points, 1 point per line
        for point in S
            point_ = [x for x in point]
            println(io, join(point_, " "))
        end
    end

    return nothing
end
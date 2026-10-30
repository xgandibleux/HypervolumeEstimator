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

For the improved reference point of Experiment 4, pass rp = reference_point_LB(inst)
(bound on the nadir point based on nMin, see instance_UKP.jl).
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

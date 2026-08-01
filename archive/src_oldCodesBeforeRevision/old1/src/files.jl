# ------------------------------------------------------------
# save on a file the points (values multiplied by -1) to be compliant with the format 
# of data for the code HV of Manuel

function writeOnFile_S(fname::String, S::Vector{Vector{Int}})
    open(fname, "w") do io
        for point in S
            point_neg = [-x for x in point]
            println(io, join(point_neg, " "))
        end
    end
end

function writeOnFile_S(fname::String, S::Set{Vector{Int64}})
    open(fname, "w") do io
        for point in S
            point_neg = [-x for x in point]
            println(io, join(point_neg, " "))
        end
    end
end

# ------------------------------------------------------------
# read on a file one float number representing the H measure

function read_Hmeasure(fname::String)
    open(fname, "r") do io
        measure = readline(io)
        return parse(Float64, measure)
    end
end

# ------------------------------------------------------------
# save the data describing an instance generated
function save_instance(fname::String, p::Matrix{Int64}, w::Vector{Int64}, c::Int64)
    d,n = size(p)       # number of objectives and number of variables
    open(fname, "w") do io
        write(io, string(n, " ", d, "\n")) # number of variables
        #write(io, string(d, "\n")) # number of objectives

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
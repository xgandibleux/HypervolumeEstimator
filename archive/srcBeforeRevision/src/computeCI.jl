
#using Statistics
#using Distributions

# ------------------------------------------------------------
"""
    confidence_interval(data::Vector{Float64}; confidence_level::Float64 = 0.95)

compute the confidence interval (not used)
"""
function confidence_interval(data::Vector{Float64}; confidence_level::Float64 = 0.95)
    n = length(data)
    mean_val = mean(data)
    std_dev = std(data)  # Écart-type échantillonnal (par défaut: correction Bessel)
    
    # Degrés de liberté
    df = n - 1

    # Quantile t de Student
    alpha = 1 - confidence_level
    t = quantile(TDist(df), 1 - alpha/2)

    # Marge d'erreur
    margin_error = t * (std_dev / sqrt(n))

    lower = mean_val - margin_error
    upper = mean_val + margin_error

    return lower, upper
end

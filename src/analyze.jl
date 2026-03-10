# ------------------------------------------------------------
"""
    average_value(y_pred)

compute the average value
"""
function average_value(y_pred)
    n = length(y_pred)
    somme = 0.0
    for i in 1:n
        somme += y_pred[i]
    end
    return somme / n
end


# ------------------------------------------------------------
"""
    average_absolue_error(y, y_pred)

compute the average absolue error
"""
function average_absolue_error(y, y_pred)
    n = length(y_pred)
    somme = 0.0
    for i in 1:n
        somme += (abs(y - y_pred[i]))
    end
    return somme / n
end


# ------------------------------------------------------------
"""
    average_relative_error(y, y_pred)

compute the average relative error
"""
function average_relative_error(y, y_pred)
    n = length(y_pred)
    somme = 0.0
    for i in 1:n
        somme += abs((y - y_pred[i]) / y)
    end
    return somme / n
end


"""
    combine_ci(means, lower, upper)

TBW
"""
function combine_ci(means, lower, upper)
    
    n = length(means)
    
    # facteur pour IC 95%
    z = 1.96
    
    # calcul des écarts-types estimés
    sigma = (upper .- lower) ./ (2*z)
    
    # poids = inverse variance
    weights = 1 ./ (sigma.^2)
    
    # moyenne pondérée
    mu = sum(weights .* means) / sum(weights)
    
    # variance combinée
    sigma_comb = sqrt(1 / sum(weights))
    
    # intervalle de confiance combiné
    ci_low = mu - z*sigma_comb
    ci_high = mu + z*sigma_comb
    
    return mu, ci_low, ci_high
end
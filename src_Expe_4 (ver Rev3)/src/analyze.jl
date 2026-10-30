# ------------------------------------------------------------
"""
    average_value(y_pred)

Compute the average value.
"""
function average_value(y_pred)
    n = length(y_pred)
    somme = 0.0
    for i in 1:n
        somme += y_pred[i]
    end
    return somme / n
end

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


# ------------------------------------------------------------
"""
    average_absolue_error(y, y_pred)

Compute the average absolute error.
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

Compute the average relative error.
"""
function average_relative_error(y, y_pred)
    n = length(y_pred)
    somme = 0.0
    for i in 1:n
        somme += abs((y - y_pred[i]) / y)
    end
    return somme / n
end

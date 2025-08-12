using Plots
#=
# --- Données partie 1 : d = 3 fixe ---
n_vals = [10, 25, 50, 75, 100]
exact_d3 = [0.01, 0.33, 14.48, 360.00, 2916.248]
approx_d3 = [0.18, 0.62, 1.88, 3.70, 6.404]

# --- Données partie 2 : n = 25 fixe ---
d_vals = [2, 3, 4, 5]
exact_n25 = [0.02, 0.33, 65.61, 1044.095]
approx_n25 = [0.46, 0.62, 0.78, 1.04]

# Création du premier graphique
p1 = plot(n_vals, exact_d3, label="Exact", lw=2, marker=:circle, yscale=:log10)
plot!(p1, n_vals, approx_d3, label="Approximatif", lw=2, marker=:square)
xlabel!(p1, "n (number of variables)")
ylabel!(p1, "Average elapsed time (sec.)")
title!(p1, "number of objectives d = 3")
#grid!(p1, true)

# Création du deuxième graphique
p2 = plot(d_vals, exact_n25, label="Exact", lw=2, marker=:circle, yscale=:log10)
plot!(p2, d_vals, approx_n25, label="Approximatif", lw=2, marker=:square)
xlabel!(p2, "d (number of objectives)")
ylabel!(p2, "Average elapsed time (sec.)")
title!(p2, "number of variables n = 25")
#grid!(p2, true)

# Combinaison en un seul graphique côte à côte
plot(p1, p2, layout=(1,2), size=(900,400))

savefig("viewTime2.png")
=#


# --- Données partie 1 : d = 3 fixe ---
n_vals = [10, 25, 50, 75, 100]
exact_d3 = [0.01, 0.33, 14.48, 360.00, 2916.248]
est_d3   = [0.18, 0.62, 1.88, 3.70, 6.404]
err_d3   = [0.008446, 0.006438, 0.006465, 0.007203, 0.004192]

# --- Données partie 2 : n = 25 fixe ---
d_vals = [2, 3, 4, 5]
exact_n25 = [0.02, 0.33, 65.61, 1044.095]
est_n25   = [0.46, 0.62, 0.78, 1.04]
err_n25   = [0.003227, 0.006438, 0.010150, 0.005054]

# --- Graphique 1 : Temps moyen vs n (d=3) ---
p1 = plot(n_vals, exact_d3, label="Exact", lw=2, marker=:circle, yscale=:log10)
plot!(p1, n_vals, est_d3, label="Estimation", lw=2, marker=:square)
xlabel!(p1, "n (number of variables)")
ylabel!(p1, "Average elapsed time (sec.)")
title!(p1, "d = 3")
#grid!(p1, true)

# --- Graphique 2 : Temps moyen vs d (n=25) ---
p2 = plot(d_vals, exact_n25, label="Exact", lw=2, marker=:circle, yscale=:log10)
plot!(p2, d_vals, est_n25, label="Estimation", lw=2, marker=:square)
xlabel!(p2, "d (number of objectives)")
ylabel!(p2, "Average elapsed time (sec.)")
title!(p2, "n = 25")
#grid!(p2, true)

# --- Graphique 3 : Erreur relative vs n (d=3) ---
#p3 = plot(n_vals, err_d3, label="Relative error", lw=2, marker=:circle)
#xlabel!(p3, "n (number of variables)")
#ylabel!(p3, "Average relative error")
#title!(p3, "Relative error (d = 3)")
#grid!(p3, true)

# --- Graphique 4 : Erreur relative vs d (n=25) ---
#p4 = plot(d_vals, err_n25, label="Relative error", lw=2, marker=:square)
#xlabel!(p4, "d (number of objectives)")
#ylabel!(p4, "Average relative error")
#title!(p4, "Relative error (n = 25)")
#grid!(p4, true)

# --- Figure combinée en 2x2 ---
#plot(p1, p2, p3, p4, layout=(2,2), size=(1000,800))
plot(p1, p2, layout=(1,2), size=(1000,800), legend=:topleft)
savefig("viewTime3.png")
using Plots

# Données
n = [100, 200, 300, 400, 500]  # nombre de variables
d2 = [3.42, 12.91, 28.68, 89.85, 94.72]
d3 = [7.31, 30.32, 68.62, 125.39, 187.32]
d4 = [7.92, 30.34, 79.19, 212.75, 365.4]
d5 = [7.94, 65.46, 114.37, 587.12, 709.39]

# Création du graphique
plot(n, d2, label = "d = 2", lw=2, marker=:circle)
plot!(n, d3, label = "d = 3", lw=2, marker=:square)
plot!(n, d4, label = "d = 4", lw=2, marker=:diamond)
plot!(n, d5, label = "d = 5", lw=2, marker=:utriangle)

# Personnalisation
xlabel!("Number of variables (n)")
ylabel!("Elapsed time (sec)")
title!("Evolution of elapsed time for n and d")
#grid!(true)
savefig("viewTime.png")

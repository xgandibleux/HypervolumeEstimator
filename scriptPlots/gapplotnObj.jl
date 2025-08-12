using Plots
using LaTeXStrings

# gap nObj

# d
d_vals = [2, 3, 4, 5]

# Experiment 1
H_YR1_exp1 = [17256378, 68019054958, 251237176539755, 840967649705240960]
avgH_exp1  = [17263977.7, 69133582996.5, 265062753215018.3, 943652862547081856.0]

# Experiment 2
H_YR1_exp2 = [16614446, 62899512829, 219845537584589, 682773590001546880]
avgH_exp2  = avgH_exp1  

# -------------------------
# Normalisation
# -------------------------
norm_exp1 = H_YR1_exp1 ./ avgH_exp1
norm_exp2 = H_YR1_exp2 ./ avgH_exp2

# -------------------------
# Plotting
# -------------------------
p = plot(d_vals, norm_exp1, label=L"H(Y_{R_1})"*" experiment 2.1", marker=:o, lw=2)
plot!(p, d_vals, norm_exp2, label=L"H(Y_{R_2})"*" experiment 2.2", marker=:s, lw=2)

# Horizontal line corresponding to the reference avgH
hline!([1.0], linestyle=:dash, color=:black, label=L"avg \ \tilde{H}")

xlabel!("number of objectives (d)")
ylabel!("H measured normalized")
title!("gap" * L"(avg \ \tilde{H}, H(Y_{R_1}))"*" and "*"gap" * L"(avg \ \tilde{H}, H(Y_{R_2}))")

display(p)

savefig("GapsnObj.png")

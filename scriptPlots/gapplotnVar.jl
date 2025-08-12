using Plots
using LaTeXStrings

# gap nObj

# n
n_vals = [100, 150, 200, 250]

# Experiment 1
H_YR1 = [68019054958, 208969505505, 535990669326.0, 1106774071350]
avgH_ref = [69133582996.5, 211406415493.4, 542928443689.6, 1119377775330.7]

# Experiment 2
H_YR2 = [62899512829, 165849348286, 400100983073.0, 710293419041]

# -------------------------
# Normalisation
# -------------------------
norm_exp1 = H_YR1 ./ avgH_ref
norm_exp2 = H_YR2 ./ avgH_ref

# -------------------------
# Plotting
# -------------------------
p = plot(n_vals, norm_exp1, label=L"H(Y_{R_1})"*" experiment 2.1", marker=:o, lw=2)
plot!(p, n_vals, norm_exp2, label=L"H(Y_{R_2})"*" experiment 2.2", marker=:s, lw=2)

# Horizontal line corresponding to the reference avgH
hline!([1.0], linestyle=:dash, color=:black, label=L"avg \ \tilde{H}")

# Ajuster l'axe X
xticks!((n_vals, string.(n_vals)))

xlabel!("number of variables (n)")
ylabel!("H measured normalized")
title!("gap" * L"(avg \ \tilde{H}, H(Y_{R_1}))"*" and "*"gap" * L"(avg \ \tilde{H}, H(Y_{R_2}))")

display(p)

savefig("GapsnVar.png")

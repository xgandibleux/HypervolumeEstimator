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

# CI
CIlow = [ 68878749037.7, 210653054418.6, 540785563336.0, 1114943417759.0]
CIhight = [ 69388416955.2, 212159776568.3, 545071324043.2, 1123812132902.4]

yerr_low = avgH_ref .- CIlow
yerr_high = CIhight .- avgH_ref 

# -------------------------
# Normalisation
# -------------------------
norm_exp1 = H_YR1 ./ avgH_ref
norm_exp2 = H_YR2 ./ avgH_ref

norm_yerr_low = yerr_low ./ avgH_ref
norm_yerr_high = yerr_high ./ avgH_ref


# -------------------------
# Plotting
# -------------------------
p = plot(n_vals, norm_exp1, label=L"H(Y_{A_1})"*" experiment 2.1", marker=:o, lw=2)
plot!(p, n_vals, norm_exp2, label=L"H(Y_{A_2})"*" experiment 2.2", marker=:s, lw=2)

# Horizontal line corresponding to the reference avgH
#hline!([1.0], linestyle=:dash, color=:black, label=L"avg \ \tilde{H}")

 plot!(n_vals, [1.0,1.0,1.0,1.0], yerror = (norm_yerr_low, norm_yerr_high),
         #label = "Avg Estimated ± CI", lw=2, marker=:circle, color=:red,
         linestyle=:dash, lw=2, marker=:point, markersize = 3, color=:black, label=L"avg \ \tilde{H}"*"± CI"
         #xticks = (listrndWeightsX, string.(listrndWeightsX)), xrotation = 45
    )

# Ajuster l'axe X
xticks!((n_vals, string.(n_vals)))

xlabel!("number of variables (n)")
ylabel!("H measured normalized")
title!("gap" * L"(avg \ \tilde{H}, H(Y_{A_1}))"*" and "*"gap" * L"(avg \ \tilde{H}, H(Y_{A_2}))")

display(p)

savefig("GapsnVarCI.png")

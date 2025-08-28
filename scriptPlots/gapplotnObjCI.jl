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

# CI
CIlow = [ 17220117.7, 68878749037.7, 263787913824960.0, 937217119799700096.0]
CIhight = [ 17307837.6, 69388416955.2, 266337592605076.6, 950088605294463104.0]

yerr_low = avgH_exp1 .- CIlow
yerr_high = CIhight .- avgH_exp1 

# -------------------------
# Normalisation
# -------------------------
norm_exp1 = H_YR1_exp1 ./ avgH_exp1
norm_exp2 = H_YR1_exp2 ./ avgH_exp2

norm_yerr_low = yerr_low ./ avgH_exp1
norm_yerr_high = yerr_high ./ avgH_exp1

# -------------------------
# Plotting
# -------------------------
p = plot(d_vals, norm_exp1, label=L"H(Y_{A_1})"*" experiment 2.1", marker=:o, lw=2)
plot!(p, d_vals, norm_exp2, label=L"H(Y_{A_2})"*" experiment 2.2", marker=:s, lw=2)

# Horizontal line corresponding to the reference avgH
#hline!([1.0], linestyle=:dash, color=:black, label=L"avg \ \tilde{H}")

 plot!(d_vals, [1.0,1.0,1.0,1.0], yerror = (norm_yerr_low, norm_yerr_high),
         #label = "Avg Estimated ± CI", lw=2, marker=:circle, color=:red,
         linestyle=:dash, lw=2, marker=:point, markersize = 3, color=:black, label=L"avg \ \tilde{H}"*"± CI"
         #xticks = (listrndWeightsX, string.(listrndWeightsX)), xrotation = 45
    )

xlabel!("number of objectives (d)")
ylabel!("H measured normalized")
title!("gap" * L"(avg \ \tilde{H}, H(Y_{A_1}))"*" and "*"gap" * L"(avg \ \tilde{H}, H(Y_{A_2}))")

display(p)

savefig("GapsnObjCI.png")

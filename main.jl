using Printf
using JuMP, GLPK, Gurobi # for solving MILP
using Distributions      # for computing the weights and CI (home version)
using SpecialFunctions   # for computing the estimation value
using HypothesisTests    # for computing the confidence interval (package version)
using Statistics         # for computing the confidence interval (home version)

include("src/instanceMO01UKP.jl")
include("src/solveMO01UKP.jl")
include("src/files.jl")
include("src/estimHyperVol1.jl")
include("src/analyze.jl")
include("src/computeCI.jl")

println("-"^80)


# =============================================================================
println("Setup the parameters...")
solver = GLPK.Optimizer
#solver = HiGHS.Optimizer
#solver = Gurobi.Optimizer
n = 5   # number of variables
o = 2     # number of objectives
rp = zeros(Int,o)
listrndWeights = [(100,100), (500,500), (1000,1000), (1500,1500), (2000,2000), (5000,5000), (10000,10000)]
trials = 20 # number of trials
println("  number of variables  : ", n)
println("  number of objectives : ", o)
println("  reference point      : ", rp)
println("  interval of #weights : ", listrndWeights)
println("  solver MIP invoked   : ", solver)
println("  number of trials     : ", trials)

allareH̃ = (Float64)[]
allCPUt = (Float64)[]

instanceName = "kp-" * string(n) * "-" * string(o)
open(instanceName*".res", "w") do ioAll
    write(ioAll, string(instanceName,"\n"))


    # =============================================================================
    println("\nGenerate an mo01UKP instance...")
    p, w, c = generate_MO01UKP(n,o)
    save_instance(instanceName* ".dat", p, w, c)


    # =============================================================================
    println("\nCompute S, the set of nondominated points...")
    start = time()
    S, cardS = solve_MO01UKP(solver, p, w, c)
    save_nondominatedpoints(instanceName*".yn",S)

    t_elapsedS = round(time() - start, digits=2)
    println("  |S|  = ",cardS, " ($t_elapsedS)s)")
    write(ioAll, string("|S|  = ",cardS, " ($t_elapsedS s) \n"))
    #write(ioAll, string("\n"))


    # =============================================================================
    println("\nCompute H, the hypervolume measure...")
    writeOnFile_S("HVpoints", S)
    if o == 2
        run(pipeline(`./src/hv -r "0 0" HVpoints`, stdout="HVmeasure"))
    elseif o == 3
        run(pipeline(`./src/hv -r "0 0 0" HVpoints`, stdout="HVmeasure"))
    elseif o == 4
        run(pipeline(`./src/hv -r "0 0 0 0" HVpoints`, stdout="HVmeasure"))
    elseif o == 5
        run(pipeline(`./src/hv -r "0 0 0 0 0" HVpoints`, stdout="HVmeasure"))
    elseif o == 6
        run(pipeline(`./src/hv -r "0 0 0 0 0 0" HVpoints`, stdout="HVmeasure"))
    end
    Hmeasure = read_Hmeasure("HVmeasure")
    @printf(". H(S) = %.1f\n", Hmeasure)

    write(ioAll, string("H(S) = ",Hmeasure, " \n"))
    write(ioAll, string("\n"))


    # =============================================================================
    for rndWeights in listrndWeights

        # -------------------------------------------------------------------------
        println("\nCompute H̃, the estimation of H...")
        listH̃ = (Float64)[]
        listCPUt = (Float64)[]

        for _ in 1:trials
            startH = time()
            H̃, numberOfWeights = H(solver, p,w,c, rp, rndWeights)
            t_elapsedH = round(time() - startH, digits=2)

            print("  H estimated with rp=$rp and $numberOfWeights weight: ")
            @printf(" %.1f ", round(H̃, digits=2) )
            println(" ($t_elapsedH s)")

            write(ioAll, string(numberOfWeights, " ", round(H̃, digits=2), " ",t_elapsedH, "s\n"))

            push!(listH̃, H̃)
            push!(listCPUt, t_elapsedH)
        end
        write(ioAll, string("\n"))


        # -------------------------------------------------------------------------
        # Confidence_interval with code given by AI
        #ci = confidence_interval(listH̃)

        # Confidence_interval with HypothesisTests package
        CIlow, CIHigh = confint( OneSampleTTest( listH̃ ), level=0.95, tail=:both )


        # -------------------------------------------------------------------------
        println("\nAnalyze the results...")
        avH̃ = average_value(listH̃)
        avCPUt = average_value(listCPUt)
        aaeH̃ = average_absolue_error(Hmeasure, listH̃)
        areH̃ = average_relative_error(Hmeasure, listH̃)

        @printf("  value H(S)                  = %.1f \n", Hmeasure)
        @printf("  average value H̃             = %.1f \n", avH̃)
        @printf("  average absolue error H̃     = %.1f \n", aaeH̃)
        @printf("  average relative error H̃    = %.6f \n", areH̃)
        #@printf("  confidence interval for 95%% = [%.1f, %.1f] \n", ci[1], ci[2])
        @printf("  confidence interval for 95%% = [%.1f, %.1f] \n",CIlow, CIHigh)
        @printf("  CPUt for computing S         = %.2f s\n", t_elapsedS)
        @printf("  average CPUt for H̃           = %.2f s\n", avCPUt)    
        push!(allareH̃, areH̃)

        write(ioAll, string("average value H̃             = ",avH̃, " \n"))
        write(ioAll, string("average absolue error H̃     = ",aaeH̃, " \n"))
        write(ioAll, string("average relative error H̃    = ",areH̃, " \n"))
        write(ioAll, string("confidence interval for 95% = ",CIlow, " ", CIHigh, " \n"))
        write(ioAll, string("average CPUt for H̃          = ",avCPUt, " \n\n"))

    end

end

println("\nAll average relative error H̃ = ", allareH̃)


listrndWeightsX = [100,500,1000,1500,2000,5000,10000]
using Plots
plot(listrndWeightsX, allareH̃, 
    seriestype = :line, 
    marker = :circle,
    title = string(n)*" variables | "*string(o)*" objectives | "*string(trials)*" trials",
    xlabel = "Number of weights",
    ylabel = "average relative error",
    legend = false,
    linewidth = 2,
    xticks = listrndWeightsX,
    xrotation = 45,
    show = true
)

savefig("kp-"*string(n)*"-"*string(o))
nothing

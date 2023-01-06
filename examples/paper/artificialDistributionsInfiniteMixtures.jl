using Distributions
using Random
using scBayesDeconv
using Plots
using DataFrames
using CSV

# Make distributions
dNoise = Dict()
dNoise["Normal"] = Normal(0,1)
dNoise["SkewNormal"] = SkewNormal(-1.32,1.65,7)
# dNoise["BimodalAsym"] = MixtureModel([
#                         Normal(-0.82-0.16,0.41),
#                         Normal(0.82-0.16,0.71),
#                         ],[.4,.6])
# dNoise["BimodalSym"] = MixtureModel([
#                         Normal(-0.9,0.5),
#                         Normal(0.9,0.5),
#                         ],[.5,.5])
# dNoise["TrimodalAsym"] = MixtureModel([
#                         Normal(-1.7+.08,0.65),
#                         Normal(0+.08,0.35),
#                         Normal(1.3+.08,0.35),
#                         ],[.2,.6,.2])
# dNoise["TrimodalSym"] = MixtureModel([
#                         Normal(-1.3,0.65),
#                         Normal(0,0.15),
#                         Normal(1.3,0.65),
#                         ],[.25,.5,.25])
dNoise["Student"] = TDist(3) 
# dNoise["Binomial"] = MixtureModel(
#     Normal[
#         Normal(-2,1),
#         Normal(6,1)
#     ],
#     [.5,.5]
#     )
# dNoise["Laplace"] = Laplace(0,.72) 
# dNoise["LaplaceSharp"] = MixtureModel([
#                         Laplace(0.,0.2),
#                         Laplace(0.,0.1),
#                         ],[.5,.5]);

dTarget = Dict()
dTarget["BimodalAsym"] = MixtureModel([
                        Normal(-.43,0.6),
                        Normal(1.67,0.6),
                        ],[.8,.2])
dTarget["BimodalSym"] = MixtureModel([
                        Normal(-0.9,0.5),
                        Normal(0.9,0.5),
                        ],[.5,.5])
dTarget["SkewNormal"] = SkewNormal(-1.32,1.65,7)

# Plot noise distributions
x = -10:.1:10
l = []
for (i,j) in pairs(dNoise)
    push!(l,plot(x,pdf.(j,x),title=i,linewidth=3,label=""))
end
fig = plot(l...,layout=(1,3),size=[1000,200])
savefig(fig,"Plots/Infinite_Noise.pdf")

# Plot target distributions
x = -5:.1:5
l = []
for (i,j) in pairs(dTarget)
    push!(l,plot(x,pdf.(j,x),title=i,linewidth=3,label=""))
end
fig = plot(l...,layout=(1,3),size=[1000,330])
savefig(fig,"Plots/Infinite_Target.pdf")

# Plot convolution distributions
x = -4:.1:4
bins=-4:.2:4
plots = []
snr=4
push!(plots,plot(ticks=nothing,axis=false))
for (i,dn) in pairs(dNoise)

    push!(plots,plot(x,pdf.(dn,x),title=i,linewidth=3,label=""))

end
for (j,dt) in pairs(dTarget)

    push!(plots,plot(x,pdf.(dt,x),title=j,linewidth=3,label=""))

    for (i,dn) in pairs(dNoise)

        data = rand(dn,10000)/snr+rand(dt,10000)
        push!(plots,histogram(data,bins=bins,label=""))

    end

end
fig = plot(plots...,layout=(4,4),size=(1000,800))
savefig(fig,"Plots/Infinite_Convolution.pdf")

# Compute statistics
x = -10:.1:10

resultsNoise = 0
try
    global resultsNoise = CSV.read("Tables/Infinite_Noise.csv",DataFrame)
catch
    global resultsNoise = DataFrame(:Noise=>String[],:Ratio=>Float64[],:Size=>Float64[],:MIO_mean=>Float64[],:MIO_std=>Float64[])
end

resultsDeconvolution = 0
try
    global resultsDeconvolution = CSV.read("Tables/Infinite_Deconvolution.csv",DataFrame)
catch
    global resultsDeconvolution = DataFrame(:Noise=>String[],:Target=>String[],:Ratio=>Float64[],:Size=>Float64[],:MIO_mean=>Float64[],:MIO_std=>Float64[],
    :MIO_Convolution_mean=>Float64[],:MIO_Convolution_std=>Float64[],:MIO_Neumann=>Float64[])
end

mioNeumann(x,y,dt) = 1-sum(abs.(pdf.(dt,x).-y)*(x[2]-x[1]))/2
for N in [100,1000,10000]
    for snr in [10,1]
        for (i,dn) in pairs(dNoise)

            n = reshape(rand(dn,N)/snr,N,1)
            fitNoise = infiniteGaussianMixture(n,κ0=0.001,k=1)

            if snr == 1 && (size(resultsNoise[(resultsNoise[!,"Size"].==N) .& (resultsNoise[!,"Noise"].==i),:])[1] == 0)
                println("Hola")
                fn(x) = pdf(dn,x[1])
                mios = scBayesDeconv.metrics.mio(fitNoise,fn)

                append!(resultsNoise,
                    ["Noise"=>i,"Ratio"=>snr,"Size"=>N,"MIO_mean"=>round(mean(mios),digits=4),"MIO_std"=>round(std(mios),digits=4)]
                )
                CSV.write("Tables/Infinite_Noise.csv", string.(resultsNoise))
            end

            for (j,dt) in pairs(dTarget)

                if size(resultsDeconvolution[(resultsDeconvolution[!,"Size"].==N) .& 
                                                (resultsDeconvolution[!,"Ratio"].==snr) .& 
                                                (resultsDeconvolution[!,"Noise"].==i) .& 
                                                (resultsDeconvolution[!,"Target"].==j),:])[1] == 0

                    println("Hola2")

                    t = reshape(rand(dt,N),N,1)
                    c = reshape(rand(dn,N)/snr.+rand(dt,N),N,1)
                    
                    ft(x) = pdf(dt,x[1])

                    fitDeconvolution = infiniteGaussianMixtureDeconvolution(c,fitNoise,k=1)
                    mios = scBayesDeconv.metrics.mio(fitDeconvolution,ft)

                    fitConvolution = infiniteGaussianMixture(c,k=1)
                    miosConvolution = scBayesDeconv.metrics.mio(fitConvolution,ft)

                    yNeumann, xNeumann = neumannDeconvolution(n,c)
                    mioneumann = mioNeumann(xNeumann,yNeumann,dt)

                    append!(resultsDeconvolution,
                        ["Noise"=>i,"Target"=>j,"Ratio"=>snr,"Size"=>N,"MIO_mean"=>round(mean(mios),digits=4),"MIO_std"=>round(std(mios),digits=4),
                        "MIO_Convolution_mean"=>round(mean(miosConvolution),digits=4),"MIO_Convolution_std"=>round(std(miosConvolution),digits=4),"MIO_Neumann"=>round(mioneumann,digits=4)]
                    )
                    CSV.write("Tables/Infinite_Deconvolution.csv", string.(resultsDeconvolution))

                end

            end

        end
    end
end

colormap = Dict([100=>"deepskyblue",1000=>"darkorange",10000=>"chartreuse3"])
stylemap = Dict([10=>:square,1=>:circle])

# Plots FFT Bayes
p1 = plot([0,1],[0,1],color="black",label="")
color = [colormap[i] for i in resultsDeconvolution[!,"Size"]]
style = [stylemap[i] for i in resultsDeconvolution[!,"Ratio"]]
scatter!(p1,resultsDeconvolution[!,"MIO_mean"],
        resultsDeconvolution[!,"MIO_Neumann"],
        color=color,markershape=style,
        xlabel="MIO Bayesian Deconvolution",
        ylabel="MIO FFT Deconvolution",
        label="")
xlims!(0,1)
ylims!(0,1)

fig = plot(p1, size=(500,500))
savefig(fig,"Plots/Infinite_FFT_Bayes.pdf")

# Plots Conv Bayes
p1 = plot([0,1],[0,1],color="black",label="")
color = [colormap[i] for i in resultsDeconvolution[!,"Size"]]
style = [stylemap[i] for i in resultsDeconvolution[!,"Ratio"]]
scatter!(p1,resultsDeconvolution[!,"MIO_mean"],
        resultsDeconvolution[!,"MIO_Convolution_mean"],
        color=color,markershape=style,
        xlabel="MIO Bayesian Deconvolution",
        ylabel="MIO Bayesian Convolution",
        label="")
xlims!(0,1)
ylims!(0,1)

fig = plot(p1, size=(500,500))
savefig(fig,"Plots/Infinite_Bayes_Conv.pdf")

# Plots FFT Conv
p1 = plot([0,1],[0,1],color="black",label="")
color = [colormap[i] for i in resultsDeconvolution[!,"Size"]]
style = [stylemap[i] for i in resultsDeconvolution[!,"Ratio"]]
scatter!(p1,resultsDeconvolution[!,"MIO_Neumann"],
        resultsDeconvolution[!,"MIO_Convolution_mean"],
        color=color,markershape=style,
        xlabel="MIO FFT Deconvolution",
        ylabel="MIO Bayesian Convolution",
        label="")
xlims!(0,1)
ylims!(0,1)

fig = plot(p1, size=(500,500))
savefig(fig,"Plots/Infinite_FFT_Conv.pdf")

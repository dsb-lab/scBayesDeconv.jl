using Distributions
using Random
using scBayesDeconv
using Plots
using DataFrames
using CSV

# Make distributions
dNoise = Dict()
dNoise["Normal"] = Normal(0,1)
# dNoise["SkewNormal"] = SkewNormal(-1.32,1.65,7)
# dNoise["Student"] = TDist(3) 

dTarget = Dict()
dTarget["BimodalAsym"] = MixtureModel([
                        Normal(-.43,0.6),
                        Normal(1.67,0.6),
                        ],[.8,.2])
# dTarget["BimodalSym"] = MixtureModel([
#                         Normal(-0.9,0.5),
#                         Normal(0.9,0.5),
#                         ],[.5,.5])
# dTarget["SkewNormal"] = SkewNormal(-1.32,1.65,7)

# Compute statistics
x = -10:.1:10

resultsNoise = DataFrame(:Noise=>String[],:Ratio=>Float64[],:Size=>Float64[],:MIO_mean=>Float64[],:MIO_std=>Float64[])

resultsDeconvolution = DataFrame(:Noise=>String[],:Target=>String[],:Ratio=>Float64[],:Size=>Float64[],:MIO_mean=>Float64[],:MIO_std=>Float64[],
    :MIO_Convolution_mean=>Float64[],:MIO_Convolution_std=>Float64[],:MIO_Neumann=>Float64[])

mioNeumann(x,y,dt) = 1-sum(abs.(pdf.(dt,x).-y)*(x[2]-x[1]))/2
for N in [1000]
    for snr in [10]
        for (i,dn) in pairs(dNoise)

            n = reshape(rand(dn,N)/snr,N,1)
            fitNoise = 0
            fitNoise = finiteGaussianMixture(n,k=4,κ0=0.001)

            if snr == 1 && (size(resultsNoise[(resultsNoise[!,"Size"].==N) .& (resultsNoise[!,"Noise"].==i),:])[1] == 0)
                fn(x) = pdf(dn,x[1])
                mios = scBayesDeconv.metrics.mio(fitNoise,fn)

                append!(resultsNoise,
                    ["Noise"=>i,"Ratio"=>snr,"Size"=>N,"MIO_mean"=>round(mean(mios),digits=4),"MIO_std"=>round(std(mios),digits=4)]
                )
                CSV.write("Tables/Finite_Noise.csv", string.(resultsNoise))
            end

            for (j,dt) in pairs(dTarget)

                if true
                    t = reshape(rand(dt,N),N,1)
                    c = reshape(rand(dn,N)/snr.+rand(dt,N),N,1)
                    
                    ft(x) = pdf(dt,x[1])

                    @time fitDeconvolution = finiteGaussianMixtureDeconvolution(c,fitNoise,k=4,ignoreSteps=0,saveSteps=10,saveEach=1)
                    mios = scBayesDeconv.metrics.mio(fitDeconvolution,ft)

                    fitConvolution = finiteGaussianMixture(c,k=4)
                    miosConvolution = scBayesDeconv.metrics.mio(fitConvolution,ft)

                    dNewmann = neumannDeconvolution(n,c)
                    mioneumann = scBayesDeconv.metrics.mio(dNewmann,ft)

                    append!(resultsDeconvolution,
                        ["Noise"=>i,"Target"=>j,"Ratio"=>snr,"Size"=>N,"MIO_mean"=>round(mean(mios),digits=4),"MIO_std"=>round(std(mios),digits=4),
                        "MIO_Convolution_mean"=>round(mean(miosConvolution),digits=4),"MIO_Convolution_std"=>round(std(miosConvolution),digits=4),"MIO_Neumann"=>round(mioneumann,digits=4)]
                    )
                    CSV.write("Tables/Finite_Deconvolution.csv", string.(resultsDeconvolution))

                    #Plots
                    local x = -4:.01:4.5
                    xx = reshape(x,1,length(x))
                    
                    p1 = histogram(c,bins=-4:.1:5,title="Convolution",label="",normalize=true,color="green",ylabel="p(x)",xlabel="x")
                    plot!(p1,x,pdf(sample(fitDeconvolution,distribution=:Convolution),xx),color="red",alpha=0.2,label="BD samples")
                    for i in 2:10
                        plot!(p1,x,pdf(sample(fitDeconvolution,distribution=:Convolution),xx),label="",color="red",alpha=0.1)
                    end
                    for i in 1:10
                        plot!(p1,x,pdf(sample(fitConvolution),xx),label="",color="green",alpha=0.1)
                    end
                    
                    p4 = histogram(n,bins=-4:.1:5,title="Noise",label="",normalize=true,color="green",ylabel="p(x)",xlabel="x")
                    plot!(p4,x,pdf(sample(fitNoise),xx),color="red",alpha=0.2,label="BD samples")
                    for i in 1:100
                        plot!(p4,x,pdf(sample(fitNoise),xx),label="",color="red",alpha=0.1)
                    end

                    p2 = histogram(t,bins=-4:.1:4,title="Bayesian deconvolution",normalize=true,color="lightblue",ylabel="p(x)",xlabel="x",label="Deconv. sample")
                    y = zeros(100,length(x))
                    plot!(p2,x,pdf(sample(fitDeconvolution),xx),color="red",alpha=0.2,label="BD samples")
                    for i in 1:100
                        plot!(p2,x,pdf(sample(fitDeconvolution),xx),label="",color="red",alpha=0.1)
                    end
                    plot!(p2,x,mean(y,dims=1)[1,:],label="Bayesian deconv.",color="black",legendfontsize=8)
                    xlims!(-4,4)
                    ylims!(0,.75)
                    
                    p3 = histogram(t,bins=-4:.1:4,title="FFT deconvolution",normalize=true,color="lightblue",ylabel="p(x)",xlabel="x",label="Deconv. sample")
                    plot!(p3,x,dNewmann(Vector(x)),linewidth=2,label="FFT decon.")
                    xlims!(-4,4)
                    ylims!(0,.75)

                    plot(p1,p4,p2,p3,layout=(1,4),size=[1400,220])
                    savefig(string("Finite_artificial_plots/",i,"_",j,"_",snr,"_",N,".pdf"))

                end

            end

        end
    end
end

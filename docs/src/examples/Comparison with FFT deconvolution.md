```julia
using Distributions
using Random
using scBayesDeconv
using Plots
using DataFrames
using CSV
```

## Create samples


```julia
N = 2000
snr = 2
dn = Normal(0,1)
dt = MixtureModel([
    Normal(-.43,0.6),
    Normal(1.67,0.6),
    ],[.8,.2])

noise = rand(dn,N)/snr
target = rand(dt,N)
convolution = rand(dn,N)/snr+rand(dt,N);
```

## Fit deconvolutions


```julia
yNeumann,xNeumann = neumannDeconvolution(noise,convolution);
```


```julia
dnfitted = infiniteGaussianMixture(reshape(noise,length(noise),1))
dtfitted = infiniteGaussianMixtureDeconvolution(reshape(convolution,length(convolution),1),dnfitted);
```

## Plot results


```julia
p1 = histogram(convolution,bins=-4:.1:5,title="Convolution",label="",normalize=true,color="green",ylabel="p(x)",xlabel="x")
histogram!(noise,bins=-2:0.1:2,inset=(1,bbox(.7,.15,.25,.4)),subplot=2,bg_inside=nothing,label="",normalize=true,title="Autofluorescence",titlefontsize=8,color="magenta",ylabel="p(x)",xlabel="x")

p2 = histogram(target,bins=-4:.1:4,title="Bayesian deconvolution",normalize=true,color="lightblue",ylabel="p(x)",xlabel="x",label="Deconv. sample")
x = -4:.01:4
y = zeros(100,length(x))
plot!(p2,x,pdf(dtfitted.samples[2],reshape(x,1,length(x))),color="red",alpha=0.2,label="BD samples")
for i in 2:100
    plot!(p2,x,pdf(dtfitted.samples[i],reshape(x,1,length(x))),label="",color="red",alpha=0.1)
    y[i,:] = pdf(dtfitted.samples[i],reshape(x,1,length(x)))[:,1]
end
plot!(p2,x,mean(y,dims=1)[1,:],label="Bayesian deconv.",color="black",legendfontsize=8)
xlims!(-4,4)

p3 = histogram(target,bins=-4:.1:4,title="FFT deconvolution",normalize=true,color="lightblue",ylabel="p(x)",xlabel="x",label="Deconv. sample")
plot!(p3,xNeumann,yNeumann,linewidth=2,label="FFT decon.")
xlims!(-4,4)

plot(p1,p2,p3,layout=(1,3),size=[1100,220])
```




    
![svg](Comparison%20with%20FFT%20deconvolution_files/Comparison%20with%20FFT%20deconvolution_7_0.svg)
    



## Evaluate metrics


```julia
f(x) = pdf(dt,x[1])
mios = scBayesDeconv.metrics.mio(dtfitted,f)

println("Bayesian MIO: ",round(mean(mios),digits=2),"±",round(std(mios),digits=2))

mioNeumann(x,y,dt) = 1-sum(abs.(pdf.(dt,x).-y)*(x[2]-x[1]))/2
println("FFT MIO: ",round(mioNeumann(xNeumann,yNeumann,dt),digits=2))
```

    Bayesian MIO: 0.94±0.02
    FFT MIO: 0.89


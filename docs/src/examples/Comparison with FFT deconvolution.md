# Comparison with FFT deconvolution methods

In this scrip we compare our method with the proposed method of Neumann [Neumann & Hössjer](https://www.tandfonline.com/doi/abs/10.1080/10485259708832708) for the deconvolution of distributions.

This method, as proposed is one dimensional altough it could be in general extended to many dimensions. In the following we will see the artifacts that this method generates.


```julia
using Distributions
using Random
using Plots
using scBayesDeconv
```

    ┌ Info: Precompiling scBayesDeconv [ba4b0364-d62a-4552-92b1-eb0a52360a94]
    └ @ Base loading.jl:1342


## Create distributions and samples


```julia
N = 5000
snr = 2
dn = Normal(0,1)
dt = MixtureModel([
    Normal(-.43,0.6),
    Normal(1.67,0.6),
    ],[.8,.2])

noise = rand(dn,N)/snr
target = rand(dt,N)
convolution = rand(dn,N)/snr+rand(dt,N);

# reshape data so it has the required shape of (samples,dimensions)
n = reshape(noise,N,1)
t = reshape(target,N,1)
c = reshape(convolution,N,1);
```

## Fit deconvolutions

We first deconvolve using the Neumann FFT method.


```julia
tNeumann = neumannDeconvolution(n,c);
```

And then the Bayesian method.


```julia
dnfitted = infiniteGaussianMixture(n)
dtfitted = infiniteGaussianMixtureDeconvolution(c,dnfitted);
```

## Plot results

As we can see the FFT method give results that have some problems as:

 - It allows the deconvolved distribution to have negative values
 - The results are somehow wavy due to the fourier basis employed.


```julia
x = -4:.01:4.5
xx = reshape(x,1,length(x))

p1 = histogram(convolution,bins=-4:.1:5,title="Convolution",label="",normalize=true,color="green",ylabel="p(x)",xlabel="x")
histogram!(noise,bins=-2:0.1:2,inset=(1,bbox(.7,.15,.25,.4)),subplot=2,bg_inside=nothing,label="",normalize=true,title="Autofluorescence",titlefontsize=8,color="magenta",ylabel="p(x)",xlabel="x")

p2 = histogram(target,bins=-4:.1:4,title="Bayesian deconvolution",normalize=true,color="lightblue",ylabel="p(x)",xlabel="x",label="Deconv. sample")
y = zeros(100,length(x))
plot!(p2,x,pdf(dtfitted.samples[2],xx),color="red",alpha=0.2,label="BD samples")
for i in 2:100
    plot!(p2,x,pdf(sample(dtfitted),xx),label="",color="red",alpha=0.1)
    y[i,:] = pdf(sample(dtfitted),xx)[:,1]
end
plot!(p2,x,mean(y,dims=1)[1,:],label="Bayesian deconv.",color="black",legendfontsize=8)
xlims!(-4,4)
ylims!(0,.75)

p3 = histogram(target,bins=-4:.1:4,title="FFT deconvolution",normalize=true,color="lightblue",ylabel="p(x)",xlabel="x",label="Deconv. sample")
plot!(p3,x,tNeumann(Vector(x)),linewidth=2,label="FFT decon.")
xlims!(-4,4)
ylims!(0,.75)

plot(p1,p2,p3,layout=(1,3),size=[1100,220])
```




    
![svg](Comparison%20with%20FFT%20deconvolution_files/Comparison%20with%20FFT%20deconvolution_9_0.svg)
    



## Evaluate metrics

To make a more rigourous comparison, we can compare the target data distribution to the deconvolution results using evaluation metrics already implemented in scBayesPackage.


```julia
f(x) = pdf(dt,x[1])
mios = scBayesDeconv.metrics.mio(dtfitted,f)
mioNeumann = scBayesDeconv.metrics.mio(tNeumann,f)

println("Bayesian MIO: ",round(mean(mios),digits=2),"±",round(std(mios),digits=2))

println("FFT MIO: ",round(mioNeumann,digits=2))
```

    Bayesian MIO: 0.97±0.02
    FFT MIO: 0.83


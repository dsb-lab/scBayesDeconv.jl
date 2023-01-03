```julia
using Distributions
using Random
using scBayesDeconv
using Plots
using MAT
```

    ┌ Info: Precompiling scBayesDeconv [ba4b0364-d62a-4552-92b1-eb0a52360a94]
    └ @ Base loading.jl:1342



```julia
N = 10000

dataActChiDye = matread("Data/Real_2_dye/20200716_FITCA.mat")["intensity"][1:N,0+1]
dataActChi = matread("Data/Real_2_dye/20200716_FITCA.mat")["intensity"][1:N,1+1]
dataChiDye = matread("Data/Real_2_dye/20200716_FITCA.mat")["intensity"][1:N,2+1]
dataChi = matread("Data/Real_2_dye/20200716_FITCA.mat")["intensity"][1:N,3+1]
dataDye = matread("Data/Real_2_dye/20200716_FITCA.mat")["intensity"][1:N,4+1]
data = matread("Data/Real_2_dye/20200716_FITCA.mat")["intensity"][1:N,5+1]

dataActChiDye2 = matread("Data/Real_2_dye/20200716_FITCA.mat")["intensity"][1:N,6+1]
dataActChi2 = matread("Data/Real_2_dye/20200716_FITCA.mat")["intensity"][1:N,7+1]
dataChiDye2 = matread("Data/Real_2_dye/20200716_FITCA.mat")["intensity"][1:N,8+1]
dataChi2 = matread("Data/Real_2_dye/20200716_FITCA.mat")["intensity"][1:N,9+1]
dataDye2 = matread("Data/Real_2_dye/20200716_FITCA.mat")["intensity"][1:N,1+10]
data2 = matread("Data/Real_2_dye/20200716_FITCA.mat")["intensity"][1:N,1+11];
```


```julia
bins = 0:100:10000

p1 = histogram(dataActChi,bins=bins,label="sample 1",title="ActChi+Auto")
histogram!(p1,dataActChi2,bins=bins,label="sample 2",alpha=0.5)

p2 = histogram(dataActChiDye,bins=bins,label="sample 1",title="ActChi+Auto+Dye")
histogram!(p2,dataActChiDye2,bins=bins,label="sample 2",alpha=0.5)

bins = 0:50:3000

p3 = histogram(dataChi,bins=bins,label="sample 1",title="Chi+Auto")
histogram!(p3,dataChi2,bins=bins,label="sample 2",alpha=0.5)

p4 = histogram(dataChiDye,bins=bins,label="sample 1",title="Chi+Auto+Dye")
histogram!(p4,dataChiDye2,bins=bins,label="sample 2",alpha=0.5)

bins = 0:50:3000

p5 = histogram(data,bins=bins,label="sample 1",title="Auto")
histogram!(p5,data2,bins=bins,label="sample 2",alpha=0.5)

p6 = histogram(dataDye,bins=bins,label="sample 1",title="Auto+Dye")
histogram!(p6,dataDye2,bins=bins,label="sample 2",alpha=0.5)

plot(p1,p2,p3,p4,p5,p6,layout=(3,2),size=[800,500])
```




    
![svg](Real%20data%20iterated%20deconvolution_files/Real%20data%20iterated%20deconvolution_2_0.svg)
    




```julia
N = 1000
```




    1000




```julia
dn = infiniteGaussianMixture(reshape(data[1:N],N,1));
dt = infiniteGaussianMixtureDeconvolution(reshape(dataDye[1:N],N,1),dn);
```


```julia
x = -200:10:5000
bins = range(-250,5000,step=30)

#Noise
n = data[1:N]
l1 = histogram(vec(n),normalize=true,bins=bins,label="noise",title="Auto (Noise)",color="magenta")
y = zeros(99,length(x))
yaux = pdf(dn.samples[1],reshape(x,1,length(x)))[:,1]
plot!(l1,x,yaux,label="Bayesian samples",color="red",alpha=.2)
for k in 2:100
    yaux = pdf(dn.samples[k],reshape(x,1,length(x)))[:,1]
    plot!(l1,x,yaux,label=nothing,color="red",alpha=.2)
    y[k-1,:] = yaux
end
plot!(l1,x,mean(y,dims=1)[1,:],label="Bayesian mean fitted",color="black",legendfontsize=8)

#Target
l2 = plot(title="Dye (Target)")#histogram(vec(t),normalize=true,bins=bins,label="target",title="ChannelB (Target)",color="green")
y = zeros(99,length(x))
yaux = pdf(dt.samples[1],reshape(x,1,length(x)))[:,1]
plot!(l2,x,yaux,label="Bayesian samples",color="red",alpha=.2)
for k in 2:100
    yaux = pdf(dt.samples[k],reshape(x,1,length(x)))[:,1]
    plot!(l2,x,yaux,label=nothing,color="red",alpha=.2)
    y[k-1,:] = yaux
end
plot!(l2,x,mean(y,dims=1)[1,:],label="Bayesian mean deconv.",color="black",legendfontsize=8)
ylims!(0,.003)

#Convolution
c = dataDye[1:N]
l3 = histogram(c,normalize=true,bins=bins,label="convolution",title=string("Auto+Dye (Convolution)"),color="lightblue")
y = zeros(99,length(x))
yaux = pdf(scBayesDeconv.sample(dt,distribution=:Convolution),reshape(x,1,length(x)))[:,1]
plot!(l3,x,yaux,label="Bayesian samples",color="red",alpha=.2)
for k in 2:100
    yaux = pdf(scBayesDeconv.sample(dt,distribution=:Convolution),reshape(x,1,length(x)))[:,1]
    plot!(l3,x,yaux,label=nothing,color="red",alpha=.2)
    y[k-1,:] = yaux
end
plot!(l3,x,mean(y,dims=1)[1,:],label="Bayesian mean deconv.",color="black",legendfontsize=8)

plot(l3,l1,l2,layout=(3,1),size=[2000,1000])
```




    
![svg](Real%20data%20iterated%20deconvolution_files/Real%20data%20iterated%20deconvolution_5_0.svg)
    




```julia
dye = [rand(scBayesDeconv.sample(dt))[1] for i in 1:N]

#dnn = infiniteGaussianMixture(reshape(dye,N,1))
dtt = infiniteGaussianMixtureDeconvolution(reshape(dataChiDye[1:N],N,1),dt);
```


```julia
x = -200:10:5000
bins = range(-250,5000,step=30)

#Noise
n = dye
l1 = histogram(vec(n),normalize=true,bins=bins,label="noise",title="Dye (Noise)",color="magenta")
y = zeros(99,length(x))
yaux = pdf(dt.samples[1],reshape(x,1,length(x)))[:,1]
plot!(l1,x,yaux,label="Bayesian samples",color="red",alpha=.2)
for k in 2:100
    yaux = pdf(dt.samples[k],reshape(x,1,length(x)))[:,1]
    plot!(l1,x,yaux,label=nothing,color="red",alpha=.2)
    y[k-1,:] = yaux
end
plot!(l1,x,mean(y,dims=1)[1,:],label="Bayesian mean fitted",color="black",legendfontsize=8)

#Target
t = dataChi[1:N]
l2 = histogram(vec(t),normalize=true,bins=bins,label="target",title="Chi+Auto (Target)",color="green")
y = zeros(99,length(x))
yaux = pdf(dtt.samples[1],reshape(x,1,length(x)))[:,1]
plot!(l2,x,yaux,label="Bayesian samples",color="red",alpha=.2)
for k in 2:100
    yaux = pdf(dtt.samples[k],reshape(x,1,length(x)))[:,1]
    plot!(l2,x,yaux,label=nothing,color="red",alpha=.2)
    y[k-1,:] = yaux
end
plot!(l2,x,mean(y,dims=1)[1,:],label="Bayesian mean deconv.",color="black",legendfontsize=8)
ylims!(0,.003)

#Convolution
c = dataChiDye[1:N]
l3 = histogram(c,normalize=true,bins=bins,label="convolution",title=string("Chi+Auto+Dye (Convolution)"),color="lightblue")
y = zeros(99,length(x))
yaux = pdf(scBayesDeconv.sample(dtt,distribution=:Convolution),reshape(x,1,length(x)))[:,1]
plot!(l3,x,yaux,label="Bayesian samples",color="red",alpha=.2)
for k in 2:100
    yaux = pdf(scBayesDeconv.sample(dtt,distribution=:Convolution),reshape(x,1,length(x)))[:,1]
    plot!(l3,x,yaux,label=nothing,color="red",alpha=.2)
    y[k-1,:] = yaux
end
plot!(l3,x,mean(y,dims=1)[1,:],label="Bayesian mean deconv.",color="black",legendfontsize=8)

plot(l3,l1,l2,layout=(3,1),size=[2000,1000])
```




    
![svg](Real%20data%20iterated%20deconvolution_files/Real%20data%20iterated%20deconvolution_7_0.svg)
    




```julia

```

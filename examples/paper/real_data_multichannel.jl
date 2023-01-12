using Distributions
using Random
using scBayesDeconv
using Plots
using Plots.Measures
using MAT
using Statistics

#Load data
channelsNames = readdir("Data/Real_1_multichannel")
channelsN = length(channelsNames)
channels = []
for file in channelsNames
    push!(channels,copy(matread(string("Data/Real_1_multichannel/",file))))
end
N = 1000#length(channels[1]["intensity"][:,1]);

#make matrices
c = zeros(N,channelsN)

for i in 1:channelsN
    c[:,i] = channels[i]["intensity"][1:N,1]
end

n = zeros(N,channelsN)

for i in 1:channelsN
    n[:,i] = channels[i]["intensity"][1:N,6]
end

#deconvolve data
dn = finiteGaussianMixture(reshape(n,N,channelsN),k=4);
dt = finiteGaussianMixtureDeconvolution(reshape(c,N,channelsN),dn,k=4);

#makePlots
figs = []

Xfitted = zeros(1000,channelsN)
for i in 1:1000
    Xfitted[i,:] .= rand(sample(dt))[:,1]
end
Xconvolved = zeros(1000,channelsN)
for i in 1:1000
    Xconvolved[i,:] .= rand(sample(dt,distribution=:Convolution))[:,1]
end
Xnoise = zeros(1000,channelsN)
for i in 1:1000
    Xnoise[i,:] .= rand(sample(dt,distribution=:Noise))[:,1]
end
for i in 1:channelsN
    for j in 1:channelsN
        x = c[1:1000,j]
        y = c[1:1000,i]
        xfitted = Xfitted[1:1000,j]
        yfitted = Xfitted[1:1000,i]
        xmax = max(sort(x)[950],sort(xfitted)[950])
        ymax = max(sort(y)[950],sort(yfitted)[950])
    
        xmin = min(sort(x)[30],sort(xfitted)[30])
        ymin = min(sort(y)[30],sort(yfitted)[30])  
        if j>i
            x = n[1:1000,j]
            y = n[1:1000,i]

            xfitted = Xnoise[1:1000,j]
            yfitted = Xnoise[1:1000,i]

            xmax = max(sort(x)[980],sort(xfitted)[980])
            ymax = max(sort(y)[980],sort(yfitted)[980])
        
            xmin = min(sort(x)[30],sort(xfitted)[30])
            ymin = min(sort(y)[30],sort(yfitted)[30])  
            push!(figs,deepcopy(scatter(x,y,markerstrokewidth=0.00,color="orange",label="")))
            scatter!(figs[end],xfitted,yfitted,markerstrokewidth=0.00,color="red",alpha=0.7,label="")
            #Put lims
            # xlims!(figs[end],xmin,xmax)
            # ylims!(figs[end],ymin,ymax)
        elseif i == j

            x = c[:,j]
            bins = range(xmin,xmax,length=20)
            push!(figs,deepcopy(histogram(sort(x)[1:990],bins=bins,alpha=0.5,
                        title=split(split(channelsNames[i],"_")[2],".")[1],
                        titlefontsize=60,
                        color="lightgreen",
                        label="",
                        normalize=true)))

            x = Xconvolved[:,j]
            histogram!(figs[end],sort(x)[1:990],bins=bins,alpha=0.5,
                        color="darkgreen",
                        label="",
                        normalize=true)

            x = Xfitted[:,j]
            histogram!(figs[end],sort(x)[1:990],bins=bins,alpha=0.5,
                        color="lightblue",
                        label="",
                        normalize=true)
        else
            x = c[1:1000,i]
            y = c[1:1000,j]

            xfitted = Xfitted[1:1000,i]
            yfitted = Xfitted[1:1000,j]

            xmax = max(sort(x)[980],sort(xfitted)[980])
            ymax = max(sort(y)[980],sort(yfitted)[980])
        
            xmin = min(sort(x)[30],sort(xfitted)[30])
            ymin = min(sort(y)[30],sort(yfitted)[30])  

            push!(figs,deepcopy(scatter(x,y,markerstrokewidth=0.00,color="lightgreen",label="")))

            x = Xfitted[1:1000,i]
            y = Xfitted[1:1000,j]
            scatter!(figs[end],x,y,markerstrokewidth=0.00,color="lightblue",label="")

            x = Xconvolved[1:1000,i]
            y = Xconvolved[1:1000,j]
            scatter!(figs[end],x,y,markerstrokewidth=0.00,color="darkgreen",label="")    
            #Put lims
            xlims!(figs[end],xmin,xmax)
            ylims!(figs[end],ymin,ymax)
        end
        plot!(figs[end],margin=10mm,xtickfontsize=20,ytickfontsize=20,xrotation=45,guidefontsize=:left,legendfontsize=30,left_margin=10mm)
    end
end
plot(figs...,layout=(channelsN,channelsN),size=[14000,14000])
savefig(string("Plots/Real_multichannel.pdf"))

#makePlots
figs = []

Xfitted = zeros(1000,channelsN)
for i in 1:1000
    Xfitted[i,:] .= rand(sample(dt))[:,1]
end
Xconvolved = zeros(1000,channelsN)
for i in 1:1000
    Xconvolved[i,:] .= rand(sample(dt,distribution=:Convolution))[:,1]
end
Xnoise = zeros(1000,channelsN)
for i in 1:1000
    Xnoise[i,:] .= rand(sample(dt,distribution=:Noise))[:,1]
end
for i in [2]
    for j in [1]
        x = c[1:1000,j]
        y = c[1:1000,i]
        xfitted = Xfitted[1:1000,j]
        yfitted = Xfitted[1:1000,i]
        xmax = max(sort(x)[950],sort(xfitted)[950])
        ymax = max(sort(y)[950],sort(yfitted)[950])
    
        xmin = min(sort(x)[30],sort(xfitted)[30])
        ymin = min(sort(y)[30],sort(yfitted)[30])  
        if j>i
            x = n[1:1000,j]
            y = n[1:1000,i]

            xfitted = Xnoise[1:1000,j]
            yfitted = Xnoise[1:1000,i]

            xmax = max(sort(x)[980],sort(xfitted)[980])
            ymax = max(sort(y)[980],sort(yfitted)[980])
        
            xmin = min(sort(x)[30],sort(xfitted)[30])
            ymin = min(sort(y)[30],sort(yfitted)[30])  
            push!(figs,deepcopy(scatter(x,y,markerstrokewidth=0.00,color="orange",label="")))
            scatter!(figs[end],xfitted,yfitted,markerstrokewidth=0.00,color="red",alpha=0.7,label="")
            #Put lims
            # xlims!(figs[end],xmin,xmax)
            # ylims!(figs[end],ymin,ymax)
        elseif i == j

            x = c[:,j]
            bins = range(xmin,xmax,length=20)
            push!(figs,deepcopy(histogram(sort(x)[1:990],bins=bins,alpha=0.5,
                        title=split(split(channelsNames[i],"_")[2],".")[1],
                        titlefontsize=60,
                        color="lightgreen",
                        label="",
                        normalize=true)))

            x = Xconvolved[:,j]
            histogram!(figs[end],sort(x)[1:990],bins=bins,alpha=0.5,
                        color="darkgreen",
                        label="",
                        normalize=true)

            x = Xfitted[:,j]
            histogram!(figs[end],sort(x)[1:990],bins=bins,alpha=0.5,
                        color="lightblue",
                        label="",
                        normalize=true)
        else
            x = c[1:1000,i]
            y = c[1:1000,j]

            xfitted = Xfitted[1:1000,i]
            yfitted = Xfitted[1:1000,j]

            xmax = max(sort(x)[980],sort(xfitted)[980])
            ymax = max(sort(y)[980],sort(yfitted)[980])
        
            xmin = min(sort(x)[30],sort(xfitted)[30])
            ymin = min(sort(y)[30],sort(yfitted)[30])  

            push!(figs,deepcopy(scatter(x,y,markerstrokewidth=0.00,color="lightgreen",label="")))

            x = Xfitted[1:1000,i]
            y = Xfitted[1:1000,j]
            scatter!(figs[end],x,y,markerstrokewidth=0.00,color="lightblue",label="")

            x = Xconvolved[1:1000,i]
            y = Xconvolved[1:1000,j]
            scatter!(figs[end],x,y,markerstrokewidth=0.00,color="darkgreen",label="")    
            #Put lims
            xlims!(figs[end],xmin,xmax)
            ylims!(figs[end],ymin,ymax)
        end
        println(channelsNames)
        plot!(figs[end],margin=10mm,xtickfontsize=10,ytickfontsize=10,xrotation=45,guidefontsize=:left,legendfontsize=30,left_margin=10mm)
            # ,xlabel=split(split(channelsNames[2],"_")[2],".")[1],ylabel=split(split(channelsNames[1],"_")[2],".")[1])
    end
end
fig = plot(figs...,layout=(1,1),size=[500,500])
savefig(fig,string("Plots/Figure5.pdf"))
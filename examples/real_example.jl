using FlowCytometry
using Plots
using DataFrames
using CSV
using Distributions
using scBayesDeconv

# #Without autofluorescence
# data = CSV.read("Data/FlowRepository_FR-FCM-Z2ST_files/attachments/fcs_control_hs1.csv",DataFrame)
# dic = Dict([string("Data/FlowRepository_FR-FCM-Z2ST_files/",i)=>string(j) for (i,j) in eachrow(data[:,["filename","dye"]])])

# fcsWOAutofluorescence = loadFCControls(dic);
# channelsCompensate = data[:,"dye"];
# Compensation.computeCompensationMatrix!(fcsWOAutofluorescence,channelsCompensate=channelsCompensate)

#With autofluorescence
data = CSV.read("Data/FlowRepository_FR-FCM-Z2ST_files/attachments/fcs_control_hs1_autofluorescence_correction.csv",DataFrame)
dic = Dict([string("Data/FlowRepository_FR-FCM-Z2ST_files/",i)=>string(j) for (i,j) in eachrow(data[:,["filename","dye"]])])

fcsWAutofluorescence = loadFCControls(dic);
channelsCompensate = data[:,"dye"];
Compensation.computeCompensationMatrix!(fcsWAutofluorescence,channelsCompensate=channelsCompensate)

# #Plotings
# f = FCSPloting.plotControls(fcsWAutofluorescence,
#     [
#         ("FITC-A","BUV563-A","BUV615-P-A"),("FITC-A","BUV563-A","BUV737-A"),
#         ("BUV496-A","BUV496-A","BUV615-P-A"),("BUV805-A","BUV805-A","BUV615-P-A"),
#         ])

# f2 = FCSPloting.plotControls(fcsWOAutofluorescence,
#     [
#         ("BUV496-A","BUV496-A","BUV615-P-A"),("BUV805-A","BUV805-A","BUV615-P-A"),
#         ])

# fs = [plot(fig,title=title) for (fig,title) in zip([f;f2],["autofluorescence","autofluorescence","with autofluorescence","with autofluorescence","without autofluorescence","without autofluorescence"])]
# fig = plot(fs..., layout = (3,2), fmt=:png, size=(1000,1000))
# savefig(fig,"correction.png")
# using JLD
# JLD.save("data.jld","a",fcsWAutofluorescence.controls["BUV563-A"].X[:,:])

X1 = fcsWAutofluorescence.controls["BUV563-A"].X[1:1000,1:10]
# X2 = JLD.load("data.jld","a")[1:1000,:]
dn = infiniteGaussianMixture(X1,k=1,α=0.00001)

data = zeros(1000,size(X1)[2])
for i in 1:1000
    data[i,:] .= rand(sample(dn))
end

for i in 1:9
    i = 1; f = scatter(X1[:,i],X1[:,i+1]); scatter!(data[:,i],data[:,i+1])
    show(f)
end
module scBayesDeconv

    #Packages
    using MLJ, Distributions, Statistics, Random, ProgressMeter

    #Models for initialization
    kmeans_ = MLJ.@load KMeans pkg=Clustering verbosity=0

    #Exported functions
    export finiteGaussianMixtureEM, finiteGaussianMixture

    #Files
    include("relabelling.jl")
    include("gaussianMixture.jl")

end

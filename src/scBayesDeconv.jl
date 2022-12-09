module scBayesDeconv

    #Packages
    using MLJ, Distributions, Statistics, Random, ProgressMeter

    #Models for initialization
    kmeans_ = MLJ.@load KMeans pkg=Clustering verbosity=0

    #Exported functions
    export GaussianFiniteMixtureModel, GaussianInfiniteMixtureModel, GaussianFiniteMixtureModelDeconvolved, GaussianInfiniteMixtureModelDeconvolved
    export finiteGaussianMixtureEM, finiteGaussianMixture, infiniteGaussianMixture, finiteGaussianMixtureDeconvolution, infiniteGaussianMixtureDeconvolution

    #Files
    include("metrics.jl")
    include("structures.jl")
    include("gaussianMixture.jl")
    include("gaussianMixtureDeconvolution.jl")
    include("newmann.jl")
    
end

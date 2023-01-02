module scBayesDeconv

    #Packages
    using MLJ, Distributions, Statistics, Random, ProgressMeter

    #Models for initialization
    kmeans_ = MLJ.@load KMeans pkg=Clustering verbosity=0

    #Exported functions
    export GaussianMixtureModel, GaussianFiniteMixtureModel, GaussianInfiniteMixtureModel, GaussianFiniteMixtureModelDeconvolved, GaussianInfiniteMixtureModelDeconvolved
    export finiteGaussianMixtureEM, finiteGaussianMixture, infiniteGaussianMixture, finiteGaussianMixtureDeconvolution, infiniteGaussianMixtureDeconvolution2, infiniteGaussianMixtureDeconvolution
    export newmannDeconvolution
    export sample
    # export MIO, MISE

    #Files
    include("structures.jl")
    include("gaussianMixture.jl")
    include("gaussianMixtureDeconvolution.jl")
    include("newmann.jl")
    include("metrics.jl")
    include("functions.jl")
    
end

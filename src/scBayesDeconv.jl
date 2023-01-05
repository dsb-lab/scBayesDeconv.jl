module scBayesDeconv

    #Packages
    using MLJ, Distributions, Statistics, Random, ProgressMeter, Interpolations, LinearAlgebra

    #Models for initialization
    kmeans_ = MLJ.@load KMeans pkg=Clustering verbosity=0

    #Exported functions
    export GaussianMixtureModel, GaussianFiniteMixtureModel, GaussianInfiniteMixtureModel, GaussianFiniteMixtureModelDeconvolved, GaussianInfiniteMixtureModelDeconvolved
    export finiteGaussianMixtureEM, finiteGaussianMixture, infiniteGaussianMixture, finiteGaussianMixtureDeconvolution, infiniteGaussianMixtureDeconvolution2, infiniteGaussianMixtureDeconvolution
    export neumannDeconvolution
    export sample
    # export MIO, MISE

    #Files
    include("structures.jl")
    include("gaussianMixture.jl")
    include("gaussianMixtureDeconvolution.jl")
    include("neumann.jl")
    include("metrics.jl")
    include("functions.jl")
    
end

abstract type GaussianMixtureModel end

mutable struct GaussianFiniteMixtureModel <: GaussianMixtureModel

    components::Int
    hyperparameters::Dict{Symbol,Any}
    likelihood::Vector{Float64}
    samples::Vector{MixtureModel}
    identities::Vector{Vector{Int}}

end

mutable struct GaussianInfiniteMixtureModel <: GaussianMixtureModel

    hyperparameters::Dict{Symbol,Any}
    samples::Vector{MixtureModel}
    identities::Vector{Vector{Int}}

end

mutable struct GaussianFiniteMixtureModelDeconvolved <: GaussianMixtureModel

    components::Int
    hyperparameters::Dict{Symbol,Any}
    samples::Vector{MixtureModel}
    identities::Vector{Vector{Int}}
    noiseDistSamples::Vector{Int}
    noiseModel::GaussianFiniteMixtureModel

end

mutable struct GaussianInfiniteMixtureModelDeconvolved <: GaussianMixtureModel

    hyperparameters::Dict{Symbol,Any}
    samples::Vector{MixtureModel}
    identities::Vector{Vector{Int}}
    noiseDistSamples::Vector{Int}
    noiseModel::GaussianMixtureModel

end
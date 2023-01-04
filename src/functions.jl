"""
    function sample(model;distribution=:Target)

Function to sample a realization from a BayesianMixtureModel.

Arguments:

 - **model::BayesianMixtureModel**: Bayesian Mixture Model from which to draw samples.

Keyword Args:

 - **distribution**: Which distribution to draw samples from.
        - :Target Samples from target distribution in MixtureModels and from the deconvolution in MixtureModelDeconvolutions.
        - :Noise Samples from noise distributinn in MixtureModelDeconvolutions.
        - :Convolution Samples from convolved distribution in MixtureModelDeconvolutions.

Return 

 A MixtureModel realization from the bayesian sampling of the model.
"""
function Distributions.sample(model::GaussianMixtureModel;distribution=:Target)

    if distribution == :Target

        N = length(model.samples)

        return model.samples[rand(1:N)]

    elseif distribution == :Noise

        N = length(model.noiseModel.samples)

        return model.noiseModel.samples[rand(1:N)]

    elseif distribution == :Convolution

        N = length(model.samples)

        mixture = MultivariateNormal[]
        w = Float64[]

        component = rand(1:N)
        dt = model.samples[component]
        dn = model.noiseModel.samples[model.noiseDistSamples[component]]

        for (wN,compN) in zip(dn.prior.p,dn.components)

            for (wT,compT) in zip(dt.prior.p,dt.components)
            
                push!(mixture,MultivariateNormal(compN.μ+compT.μ,compN.Σ+compT.Σ))
                push!(w,wN*wT)
            
            end

        end

        return MixtureModel(mixture,w)
    else

        error("Distribution can only be :Target, :Noise or :Convolution")

    end
end
function sample(model;distribution=:Target)

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
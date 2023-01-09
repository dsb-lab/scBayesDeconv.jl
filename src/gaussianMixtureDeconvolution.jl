cartesian2lin(x,y,k1,k2) = x+(y-1)*(k1)
function lin2cartesian(x,k1,k2) 
    x1 = 0
    x2 = 0
    if x%k1==0 
        x1 = k1 
    else
        x1 = x%k1
    end
    
    x2 = (x-x1)÷k1+1

    return x1,x2
end

function gmloglikelihoodConvolution!(p::Matrix,X::Matrix,centers::Vector,covariances::Vector,weights::Vector,centersN::Vector,covariancesN::Vector,weightsN::Vector)

    kT = size(centers)[1]
    kN = size(centersN)[1]
    for j in 1:kT
        for k in 1:kN
            #Make MultivariateNormal from Distributions
            μ = centers[j]+centersN[k]
            Σ = covariances[j]+covariancesN[k]
            w = weights[j]*weightsN[k]
            dist = MultivariateNormal(μ,Σ)
            #Compute the logpdf for each sample
            for i in 1:size(X)[1]
                x = @views X[i,:]
                p[i,cartesian2lin(k,j,kN,kT)] = logpdf(dist,x)+log(w)
            end
        end
    end

    return
end

function gmlikelihoodConvolution!(p::Matrix,X::Matrix,centers::Vector,covariances::Vector,weights::Vector,centersN::Vector,covariancesN::Vector,weightsN::Vector)

    gmloglikelihoodConvolution!(p, X, centers, covariances, weights, centersN, covariancesN, weightsN)

    #Rescale
    p .-= maximum(p,dims=2)
    #Exp
    p .= exp.(p)
    #Normalize
    p ./= sum(p,dims=2)

    return
end

"""
    finiteGaussianMixtureDeconvolution(X::Matrix, Y::GaussianFiniteMixtureModel;
        k::Int,
        initialization::Union{String,Matrix} = "kmeans",
        α = 1,
        ν0 = 1,
        κ0 = 0.001,
        μ0 = nothing,
        Σ0 = nothing,
        ignoreSteps::Int = 1000, 
        saveSteps::Int = 1000,
        saveEach::Int = 10,
        verbose = false,
        seed = 0
        )

Function to fit a convolved finite mixture model to data.

Arguments:

 - **X::Matrix**: Matrix with data to be fitted as (realizations, dimensions).
 - **Y::GaussianFiniteMixtureModel**: Mixture model used to fit the noise of the convolution.

Keyword Arguments:

 - **k::Int**: Number of components of the mixture of the target.
 - **initialization::Union{String,Matrix} = "kmeans"**: Method to initializate the mixture parameters. 
 - **α = 1**: Hyperparameter of the Dirichlet distribution. The higher, the more probable that a cell will be assigned to another distribution.
 - **ν0 = 1**: Hyperparameter of the InverseWishart distribution. The highler, the more wight has the pior InverseWishart.
 - **κ0 = 0.001**: Hyperparameter of the Normal prior distribution. The higher, the more focussed will be the prior Normal around the mean.
 - **μ0 = nothing**: Hyperparameter of indicating the mean of the Normal. If nothing it will be estimated.
 - **Σ0 = nothing**: Hyperparameter indicating the prior Covariance distribution of the model. If nothing it will be estimated.
 - **ignoreSteps::Int = 1000**: Number of steps to perform before saving realizations.
 - **saveSteps::Int = 1000**: Number of steps to perform from which we will save samples.
 - **saveEach::Int = 10**: Number of steps to take before saving a sample. 
 - **verbose = false**: Is to show progress of the fitting.
 - **seed = 0**: Seed of the random generator.

Return 

A GaussianFiniteMixtureModelDeconvolved with the sampling of the bayesian model.
"""
function finiteGaussianMixtureDeconvolution(X::Matrix, Y::Union{GaussianFiniteMixtureModel,GaussianFiniteMixtureModelDeconvolved};
    k::Int,
    initialization::Union{String,Matrix} = "finiteMixtureModel",
    α = 1,
    ν0 = 1,
    κ0 = 0.001,
    μ0 = nothing,
    Σ0 = nothing,
    ignoreSteps::Int = 1000, 
    saveSteps::Int = 1000,
    saveEach::Int = 10,
    verbose = false
    )

    nCells, dimensions = size(X)

    #Activate progress line
    if verbose 
        ProgressMeter.ijulia_behavior(:clear)
        prog = Progress(length(ignoreSteps+saveSteps))
    end 

    #Initialization
    μ0,Σ0 = initializationGaussianMixtureHyperparameters(X,μ0,Σ0)
    centers,covariances,weights,identities = initializationGaussianMixture(X,k,initialization,
                                                                            α,
                                                                            ν0,
                                                                            κ0,
                                                                            μ0,
                                                                            Σ0,
                                                                            ignoreSteps,
                                                                            saveSteps,
                                                                            saveEach)

    #Auxiliar functions
    kN = Y.components
    p = zeros(nCells,k*kN)
    votes = fill(0,k*kN) #Auxiliar for sampling from the Dirichlet distributions
    votesK = fill(0,k) #Auxiliar for sampling from the Dirichlet distributions
    vote = fill(0,k*kN) #Auxiliar for sampling from the Dirichlet distributions

    saveIds = Vector{Int}[]
    saveDist = MixtureModel[]
    saveN = Int[]

    nSamples = length(Y.samples)
    nRange = range(1,nSamples,step=1)
    centersN = [i.μ for i in Y.samples[1].components]
    covariancesN = [i.Σ for i in Y.samples[1].components]
    weightsN = Y.samples[1].prior.p
    identitiesN = copy(identities)

    m = zeros(dimensions)
    m2 = zeros(dimensions,dimensions)
    S2 = zeros(dimensions,dimensions)

    #Loop
    for step in 1:(ignoreSteps+saveSteps)

        #Sample noise distribution
        nSample = rand(nRange)
        centersN = [i.μ for i in Y.samples[nSample].components]
        covariancesN = [i.Σ for i in Y.samples[nSample].components]
        weightsN = Y.samples[nSample].prior.p        

        #Sample identities
        gmlikelihoodConvolution!(p,X,centers,covariances,weights,centersN,covariancesN,weightsN)
        votes .= 0 #Reset votes
        votesK .= 0 #Reset votes
        for i in 1:nCells           
            vote .= rand(Multinomial(1,@views(p[i,:])))
            pos = lin2cartesian(findfirst(vote.==1),kN,k)
            votes .+= vote
            votesK[pos[2]] += 1
            identities[i] = pos[2]
            identitiesN[i] = pos[1]
        end
        #Sample parameters
            #Sample weights
        weights .= rand(Dirichlet(votesK.+α/k))
        total = 0
        for comp in 1:k
            idsT = identities.==comp

            if votesK[comp] > dimensions #Check if we have enough statistical power to compute the wishart  

                #Sample covariance
                m2 .= 0
                S2 .= 0   
                ids = 0
                for compN in 1:kN
                    idsN = identitiesN.==compN
                    ids = idsN .& idsT
                    total += sum(ids)
                    if sum(ids) > dimensions
                        #Statistics
                        aux = (reshape(mean(X[ids,:],dims=1),dimensions)-centersN[compN]-centers[comp])
                        m2 .+= votes[cartesian2lin(compN,comp,kN,k)]*aux*transpose(aux)
                        S2 .+= votes[cartesian2lin(compN,comp,kN,k)]*(cov(@views(X[ids,:]),corrected=false)-covariancesN[compN])
                    else
                        m2 .+= 0.
                        S2 .+= 0.
                    end
                end
                # for i in 1:dimensions
                #     if S2[i,i] <= 0
                #         S2[i,i] = 0.
                #     end
                # end
                if votesK[comp] > 1 && isposdef(S2)
                    neff = votesK[comp]+ν0+1

                    Σeff = S2 + m2 + κ0*(centers[comp]-μ0)*transpose((centers[comp]-μ0)) + Σ0
                    Σeff = (Σeff+transpose(Σeff))/2 #Reinforce hermicity
                    covariances[comp] = rand(InverseWishart(neff,Σeff))
                    covariances[comp] = (covariances[comp]+transpose(covariances[comp]))/2
                end
                #Sample centers
                m .= 0
                S2 .= 0   
                for compN in 1:kN
                    idsN = identitiesN.==compN
                    ids = idsN .& idsT
                    s = inv(covariances[comp]+covariancesN[compN])
                    #Statistics
                    if sum(ids) > dimensions
                        m .+= s*(votes[cartesian2lin(compN,comp,kN,k)]*(reshape(mean(X[ids,:],dims=1),dimensions)-centersN[compN])+κ0*μ0)
                    else
                        m .+= 0
                    end
                    S2 .+= s*(votes[cartesian2lin(compN,comp,kN,k)]+κ0)
                end
                S2 = inv(S2)
                m = S2*m
                centers[comp] .= rand(MultivariateNormal(m,S2))

            end
        end

        #Save
        if step >= ignoreSteps && step%saveEach == 0
            #rel = relabeling(identities,identitiesRef)
            dist = MixtureModel(MultivariateNormal[[MultivariateNormal(copy(i),copy(j)) for (i,j) in zip(centers,covariances)]...],copy(weights))
            push!(saveDist,deepcopy(dist))
            push!(saveIds,copy(identities))
            push!(saveN,nSample)
        end

        #Show progress bar if verbose
        if verbose
            next!(prog,showvalues=[(:iter,step)])
        end

    end

    return GaussianFiniteMixtureModelDeconvolved(k,
                                Dict([  
                                        :α=>α,
                                        :ν0 => ν0,
                                        :κ0 => κ0,
                                        :μ0 => μ0,
                                        :Σ0 => Σ0
                                    ]),
                                saveDist,
                                saveIds,
                                saveN,
                                Y)
end

"""
    infiniteGaussianMixtureDeconvolution(X::Matrix, Y::GaussianMixtureModel;
        k::Int,
        initialization::Union{String,Matrix} = "kmeans",
        α = 1,
        ν0 = 1,
        κ0 = 0.001,
        μ0 = nothing,
        Σ0 = nothing,
        ignoreSteps::Int = 1000, 
        saveSteps::Int = 1000,
        saveEach::Int = 10,
        verbose = false,
        seed = 0,
        prune = 0.1
        )

Function to fit a convolved finite mixture model to data.

Arguments:

 - **X::Matrix**: Matrix with data to be fitted as (realizations, dimensions).
 - **Y::GaussianMixtureModel**: Mixture model used to fit the noise of the convolution.

Keyword Arguments:

 - **k::Int**: Number of components of the mixture to start with.
 - **initialization::Union{String,Matrix} = "kmeans"**: Method to initializate the mixture parameters. 
 - **α = 1**: Hyperparameter of the Dirichlet distribution. The higher, the more probable that a cell will be assigned to another distribution.
 - **ν0 = 1**: Hyperparameter of the InverseWishart distribution. The highler, the more wight has the pior InverseWishart.
 - **κ0 = 0.001**: Hyperparameter of the Normal prior distribution. The higher, the more focussed will be the prior Normal around the mean.
 - **μ0 = nothing**: Hyperparameter of indicating the mean of the Normal. If nothing it will be estimated.
 - **Σ0 = nothing**: Hyperparameter indicating the prior Covariance distribution of the model. If nothing it will be estimated.
 - **ignoreSteps::Int = 1000**: Number of steps to perform before saving realizations.
 - **saveSteps::Int = 1000**: Number of steps to perform from which we will save samples.
 - **saveEach::Int = 10**: Number of steps to take before saving a sample. 
 - **verbose = false**: Is to show progress of the fitting.
 - **seed = 0**: Seed of the random generator.
 - **prune = 0.1**: Cutoff to remove any basis from the noise distribution that is below this weight. 
As infinite mixtures may have many basis that are only allocated to ver few cells (starting clusters) during the creation of new clusters,
they do not realy contribute to the effective distribution but increase substantially the number of call to the likelihood function that is one of the most costly steps.
Pruning unimportant basis can highly reduce the computation of the convolved model.

Return 

A GaussianInfiniteMixtureModelDeconvolved with the sampling of the bayesian model.
"""
function infiniteGaussianMixtureDeconvolution(X::Matrix, Y::GaussianMixtureModel;
    k = 1,
    initialization::Union{String,Matrix} = "kmeans",
    α = 1,
    ν0 = size(X)[2]+1,
    κ0 = 0.001,
    μ0 = nothing,
    Σ0 = nothing,
    ignoreSteps::Int = 1000, 
    saveSteps::Int = 1000,
    saveEach::Int = 10,
    verbose = false,
    seed = 0,
    prune = 0.1
    )

    Random.seed!(seed)

    nCells, dimensions = size(X)

    #Activate progress line
    if verbose 
        ProgressMeter.ijulia_behavior(:clear)
        prog = Progress(length(ignoreSteps+saveSteps))
    end 

    #Initialization
    centers,covariances,weights,identities = initializationGaussianMixture(X,k,initialization)
    #Hyperparameters
    μ0,Σ0 = initializationGaussianMixtureHyperparameters(X,μ0,Σ0)

    #Auxiliar functions
    vote = fill(0,k) #Auxiliar for sampling from the Dirichlet distributions

    saveIds = Vector{Int}[]
    saveDist = MixtureModel[]
    saveN = Int[]

    nSamples = length(Y.samples)
    nRange = range(1,nSamples,step=1)
    centersN = [i.μ for i in Y.samples[1].components]
    covariancesN = [i.Σ for i in Y.samples[1].components]
    weightsN = Y.samples[1].prior.p
    identitiesN = copy(identities)

    kN = 1; kT = 1; kTotal = 1
    nSample = 0
    #Statistics
    m = fill(zeros(dimensions),kN,kT)
    S2 = fill(zeros(dimensions,dimensions),kN,kT)
    n = zeros(Int,kN,kT)
    #Effective parameters
    m0 = zeros(kN,kT)
    m1 = zeros(kN,kT)
    νeff = zeros(kN,kT)
    μyeff = fill(zeros(dimensions),kN,kT)
    Σyeff = fill(zeros(dimensions,dimensions),kN,kT)
    #Noise
    nSample = 0
    centersN = 0
    covariancesN = 0
    weightsN = 0      
    p = 0
    w = 0
    
    #Preallocate
    m2_ = zeros(dimensions,dimensions)
    S2_ = zeros(dimensions,dimensions)
    m_ = zeros(dimensions)
    Σeff_ = zeros(dimensions,dimensions)

    #Loop
    for step in 1:(ignoreSteps+saveSteps)

        if true#step == 1
            #Sample noise distribution
            nSample = rand(nRange);
            centersN = [i.μ for (j,i) in enumerate(Y.samples[nSample].components) if Y.samples[nSample].prior.p[j] > prune]
            covariancesN = [i.Σ for (j,i) in enumerate(Y.samples[nSample].components) if Y.samples[nSample].prior.p[j] > prune]
            weightsN = Y.samples[nSample].prior.p[Y.samples[nSample].prior.p .> prune]
            kT = length(centers)
            kTotal = kT
            kN = length(centersN)
            p = zeros(nCells,kT*kN)
            w = zeros(kT*kN+kN)
            
            #Statistics
            m = fill(zeros(dimensions),kN,kT)
            S2 = fill(zeros(dimensions,dimensions),kN,kT)
            n = zeros(Int,kN,kT)

            #Sample new noise identities
            gmlikelihoodConvolution!(p,X,centers,covariances,weights,centersN,covariancesN,weightsN)
            for i in 1:nCells           
                vote = rand(Multinomial(1,@views(p[i,:])))
                pos = lin2cartesian(findfirst(vote.==1),kN,kT)
                # identities[i] = pos[2]
                identitiesN[i] = pos[1]
                n[pos[1],identities[i]] += 1 #Keep target base, but change noise base
            end

            #Effective parameters
            m0 = zeros(kN,kT)
            m1 = zeros(kN,kT)
            νeff = zeros(kN,kT)
            μyeff = fill(zeros(dimensions),kN,kT)
            Σyeff = fill(zeros(dimensions,dimensions),kN,kT)

            #Compute statistics
            for j in 1:kN
                for k in 1:kT
                    ids = (identities .== k) .& (identitiesN .== j)
                    if sum(ids) != 0
                        m[j,k] = reshape(mean(@views(X[ids,:]),dims=1),dimensions)
                        S2[j,k] = cov(@views(X[ids,:]),dims=1,corrected=false)
                    end
                    m0[j,k] = (n[j,k]+κ0)/(n[j,k]+κ0+1)
                    m1[j,k] = ((n[j,k]*κ0)/(n[j,k]+κ0)+n[j,k]*κ0)/(n[j,k]+κ0+1)
                    νeff[j,k] = (n[j,k]+ν0+1-dimensions)
                    μyeff[j,k] = (n[j,k]*m[j,k]+κ0*μ0)/(n[j,k]+κ0)
                    aux = (m1[j,k]*(μ0-m[j,k])*transpose(μ0-m[j,k])+n[j,k]*S2[j,k]+Σ0)/m0[j,k]
                    Σyeff[j,k] = (aux+transpose(aux))/2
                end
            end

        end
        # println("S2",S2,"\n Σyeff",Σyeff,"\n m1",m1,"\n m0",m0,"\n")

        for i in 1:nCells

            id = identities[i]
            idN = identitiesN[i]

            #Remove sample
            # println(n,sum(n,dims=1),n[idN,id],"/",sum(n),(idN,id))
            if sum(n[:,id]) == 1 #Remove component
                popat!(centers,id)
                popat!(covariances,id)
                popat!(weights,id)
                m = m[:,id .!= 1:kT]
                S2 = S2[:,id .!= 1:kT]
                n = n[:,id .!= 1:kT]
                w = w[1:end-kN]
                m0 = m0[:,id .!= 1:kT]
                m1 = m1[:,id .!= 1:kT]
                νeff = νeff[:,id .!= 1:kT]
                μyeff = μyeff[:,id .!= 1:kT]
                Σyeff = Σyeff[:,id .!= 1:kT]
                identities[identities.>id] .-= 1 #Reduce index of components above the removed one
                kT -= 1
                # println("Component removed")
            else #Modify statistics
                #Statistics
                if n[idN,id] == 1
                    mnew = zeros(dimensions)
                    S2[idN,id] .= 0
                else
                    mnew = (n[idN,id]*m[idN,id]-X[i,:])/(n[idN,id]-1)
                    S2[idN,id] .= (n[idN,id]*S2[idN,id] + n[idN,id]*m[idN,id]*transpose(m[idN,id]) - X[i,:]*transpose(X[i,:]))/(n[idN,id]-1) - mnew*transpose(mnew)
                end
                m[idN,id] .= mnew
                n[idN,id] -= 1 
                #Effective parameters
                m0[idN,id] = (n[idN,id]+κ0)/(n[idN,id]+κ0+1)
                m1[idN,id] = ((n[idN,id]*κ0)/(n[idN,id]+κ0)+n[idN,id]*κ0)/(n[idN,id]+κ0+1)
                νeff[idN,id] = n[idN,id]+ν0+1-dimensions
                μyeff[idN,id] = (n[idN,id]*m[idN,id]+κ0*μ0)/(n[idN,id]+κ0)
                Σyeff[idN,id] = (m1[idN,id]*(μ0-m[idN,id])*transpose(μ0-m[idN,id])+n[idN,id]*S2[idN,id]+Σ0)/m0[idN,id]
                Σyeff[idN,id] = (Σyeff[idN,id]+transpose(Σyeff[idN,id]))/2 #Solve problem with hermitian
            end

            #Reassign sample
            w .= 0
            for j in 1:kN
                for k in 1:kT
                    try
                        w[cartesian2lin(j,k,kN,kT+1)] = logpdf(MvTDist(νeff[j,k],μyeff[j,k],Σyeff[j,k]/νeff[j,k]),@views(X[i,:]))+log(n[j,k]/(nCells+α-1))
                    catch
                        println("Asignation failled.")
                    end
                end
                w[cartesian2lin(j,kT+1,kN,kT+1)] = logpdf(MultivariateNormal(μ0+centersN[kN],Σ0+covariancesN[kN]),@views(X[i,:]))+log(α/(nCells+α-1))
            end
            w .-= maximum(w)
            w = exp.(w)
            w ./= sum(w)
            vote = rand(Multinomial(1,w))
            aux = lin2cartesian(findfirst(vote.==1),kN,kT+1)
            identities[i] = aux[2]
            identitiesN[i] = aux[1]

            #Update statistics
            id = identities[i]
            idN = identitiesN[i]
            if id == kT + 1 #Add component
                # println("Component added")
                # Add new data
                m = hcat(m,fill(X[i,:],kN,1))
                S2 = hcat(S2,fill(zeros(dimensions,dimensions),kN,1))
                n = hcat(n,fill(0,kN))
                centers = push!(centers,copy(μ0))
                covariances = push!(covariances,copy(Σ0))
                weights = push!(weights,1/nCells)
                append!(w,zeros(kN))
                m0 = hcat(m0,fill(0,kN))
                m1 = hcat(m1,fill(0,kN))
                νeff = hcat(νeff,fill(0,kN))
                μyeff = hcat(μyeff,fill(zeros(dimensions),kN,1))
                Σyeff = hcat(Σyeff,fill(zeros(dimensions,dimensions),kN,1))

                k = id
                n[idN,id] += 1
                for j in 1:kN
                    ids = (identities .== k) .& (identitiesN .== j)
                    if sum(ids) != 0
                        m[j,k] = reshape(mean(@views(X[ids,:]),dims=1),dimensions)
                        S2[j,k] = cov(@views(X[ids,:]),dims=1,corrected=false)
                    end
                    m0[j,k] = (n[j,k]+κ0)/(n[j,k]+κ0+1)
                    m1[j,k] = ((n[j,k]*κ0)/(n[j,k]+κ0)+n[j,k]*κ0)/(n[j,k]+κ0+1)
                    νeff[j,k] = (n[j,k]+ν0+1-dimensions)
                    μyeff[j,k] = (n[j,k]*m[j,k]+κ0*μ0)/(n[j,k]+κ0)
                    aux = (m1[j,k]*(μ0-m[j,k])*transpose(μ0-m[j,k])+n[j,k]*S2[j,k]+Σ0)/m0[j,k]
                    Σyeff[j,k] = (aux+transpose(aux))/2
                end
        
                # println("----S2",S2,"\n Σyeff",Σyeff,"\n m1",m1,"\n m0",m0,"\n")

                kT += 1
            else #Modify statistics
                #Statistics
                mnew = (n[idN,id]*m[idN,id]+X[i,:])/(n[idN,id]+1)
                S2[idN,id] = (n[idN,id]*S2[idN,id] + n[idN,id]*m[idN,id]*transpose(m[idN,id]) + X[i,:]*transpose(X[i,:]))/(n[idN,id]+1) - mnew*transpose(mnew)
                m[idN,id] .= mnew
                n[idN,id] += 1
                #Effective parameters
                m0[idN,id] = (n[idN,id]+κ0)/(n[idN,id]+κ0+1)
                m1[idN,id] = ((n[idN,id]*κ0)/(n[idN,id]+κ0)+n[idN,id]*κ0)/(n[idN,id]+κ0+1)
                νeff[idN,id] = n[idN,id]+ν0+1-dimensions
                μyeff[idN,id] = (n[idN,id]*m[idN,id]+κ0*μ0)/(n[idN,id]+κ0)
                Σyeff[idN,id] = (m1[idN,id]*(μ0-m[idN,id])*transpose(μ0-m[idN,id])+n[idN,id]*S2[idN,id]+Σ0)/m0[idN,id]
                Σyeff[idN,id] = (Σyeff[idN,id]+transpose(Σyeff[idN,id]))/2 #Solve problem with hermitian
            end

        end

        #Save
        if step >= ignoreSteps && step%saveEach == 0

            push!(saveN,nSample)

        #Sample parameters
            #Sample weights
            votesK = reshape(sum(n,dims=1),kT)
            weights .= rand(Dirichlet(votesK.+α/k))
            for comp in 1:kT
                idsT = identities.==comp
    
                if votesK[comp] > dimensions #Check if we have enough statistical power to compute the wishart  
    
                    #Sample covariance
                    m2_ .= 0#zeros(dimensions,dimensions)
                    S2_ .= 0#zeros(dimensions,dimensions)
                    for compN in 1:kN
                        idsN = identitiesN.==compN
                        ids = idsN .& idsT
                        #Statistics
                        if sum(ids) > 0
                            aux = (reshape(mean(X[ids,:],dims=1),dimensions)-centersN[compN]-centers[comp])
                            m2_ .+= n[compN,comp]*aux*transpose(aux)
                            S2_ .+= n[compN,comp]*(cov(@views(X[ids,:]),corrected=false)-covariancesN[compN])
                        end
                    end
                    # println(votes,m/votesK[comp],S2/votesK[comp])
                    for i in 1:dimensions
                        if S2_[i,i] < 0
                            S2_[i,i] = 0
                        end
                    end
                    neff_ = votesK[comp]+ν0+1
    
                    Σeff_ .= S2_ + m2_ + κ0*(centers[comp]-μ0)*transpose((centers[comp]-μ0)) + Σ0
                    # println(S2, m2, κ0*(centers[comp]-μ0)*transpose((centers[comp]-μ0)), Σ0)
                    Σeff_ .= (Σeff_+transpose(Σeff_))/2 #Reinforce hemicity
                    try
                        covariances[comp] = rand(InverseWishart(neff_,Σeff_))
                    catch
                        println("WARNING: Sampling has failed for this covariance matrix. In general this will not be a problem if only happens in rare ocassions.")
                    end
                    #Sample centers
                    S2_ .= 0
                    m_ .= 0
                    for compN in 1:kN
                        idsN = identitiesN.==compN
                        ids = idsN .& idsT
                        s = inv(covariances[comp]+covariancesN[compN])
                        #Statistics
                        # println((votes[cartesian2lin(compN,comp,kN,k)]*(reshape(mean(X[ids,:],dims=1),dimensions)-centersN[compN])+κ0*μ0))
                        if sum(ids) > 0
                            m_ .+= s*(n[compN,comp]*(reshape(mean(X[ids,:],dims=1),dimensions)-centersN[compN])+κ0*μ0)
                            S2_ .+= s*(n[compN,comp]+κ0)
                        end
                    end
                    S2_ .= inv(S2_)
                    S2_ .= (S2_+transpose(S2_))/2
                    m_ .= S2_*m_
                    centers[comp] .= rand(MultivariateNormal(m_,S2_))
                end
            end
    
            #rel = relabeling(identities,identitiesRef)
            dist = MixtureModel(MultivariateNormal[[MultivariateNormal(copy(i),copy(j)) for (i,j) in zip(centers,covariances)]...],copy(weights))
            push!(saveDist,deepcopy(dist))
            push!(saveIds,copy(identities))
            
        end

        #Show progress bar if verbose
        if verbose
            next!(prog,showvalues=[(:iter,step)])
        end

    end

    return GaussianInfiniteMixtureModelDeconvolved(
                                Dict([  
                                        :α=>α,
                                        :ν0 => ν0,
                                        :κ0 => κ0,
                                        :μ0 => μ0,
                                        :Σ0 => Σ0
                                    ]),
                                saveDist,
                                saveIds,
                                saveN,
                                Y
                                )

end

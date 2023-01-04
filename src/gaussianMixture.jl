function relabeling(indicator,indicatorRef)
    k = maximum(indicator)
    #Initialize
    C = zeros(k,k) #Missclassification matrix
    #Create cost matrix
    for i in 1:k
        for j in 1:k
            C[i,j] = sum((indicatorRef .== i) .& (indicator .!= j))
        end
    end
    #Perform Hungarian Algorithm
    assignement, cost = hungarian(C)
    
    return assignement
end

function closerPoints(X::Matrix,centers::Matrix)
    if size(X)[2] != size(centers)[2]
        error("Second dimension")
    end
    d = zeros(size(X)[1],size(centers)[1])

    for i in 1:size(X)[1]
        x = @views X[i,:]
        for j in 1:size(centers)[1]
            c = @views centers[j,:]
            d[i,j] = sum((x.-c).^2)
        end
    end

    return Matrix([i[2] for i in argmin(d,dims=2)])[:,1]
end

function gmloglikelihood!(p::Matrix,X::Matrix,centers::Vector,covariances::Vector,weights::Vector)

    for j in 1:size(centers)[1]
        #Make MultivariateNormal from Distributions
        μ = centers[j]
        Σ = covariances[j]
        w = weights[j]
        dist = MultivariateNormal(μ,Σ)
        #Compute the logpdf for each sample
        for i in 1:size(X)[1]
            x = @views X[i,:]
            p[i,j] = logpdf(dist,x)+log(w)
        end
    end

    return
end

function gmlikelihood!(p::Matrix,X::Matrix,centers::Vector,covariances::Vector,weights::Vector)

    gmloglikelihood!(p,X,centers,covariances,weights)

    #Rescale
    p .-= maximum(p,dims=2)
    #Exp
    p .= exp.(p)
    #Normalize
    p ./= sum(p,dims=2)

    return
end

function initializationGaussianMixture(
                                    X::Matrix,
                                    k::Int,
                                    initialization::Union{String,Vector} = "kmeans"
                                    )

    nCells, dimensions = size(X)

    #Check shape of initial centers
    if typeof(initialization) <: Vector
        if size(initialization)[1] != size(X)[1] || maximum(initialization) > k || minimum(initialization) < 1
            error("If initialization is a matrix of given the identitities, the dimensions must be the same size as samples and have identities in the range (1,k).")
        end
    end

    centers = Vector{Float32}[]
    identities = zeros(Int, size(X)[1])
    covariances = Matrix{Float32}[]
    weights = zeros(Float32, k)
    if initialization in ["kmeans","random"] 
        if initialization == "kmeans" && k > 1
            #Fit kmeans
            model = kmeans_(k=k)
            mach = machine(model,X,scitype_check_level=0)
            fit!(mach,verbosity=0)
            centers = permutedims(fitted_params(mach)[1])
        else
            centers = rand(k,size(X)[2])
            centers .= centers .* (maximum(X,dims=1).-minimum(X,dims=1)) .-minimum(X,dims=1) #Scale to be in the box
        end
        #Identities
        identities = closerPoints(X,centers)
        #Centers
        centers = [reshape(mean(X[identities.==i,:],dims=1),dimensions) for i in 1:k]
        #Covariances
        for id in 1:k
            push!(covariances,cov(@views(X[identities.==id,:]),corrected=false))
        end
        #Weights
        weights = [sum(identities.==id)/nCells for id in 1:k]
    elseif typeof(initialization) <: Vector
        #Identitites
        identities .= initialization
        #Centers
        weights = [reshape(mean(@views(X[identities.==id,:]),dims=1),dimensions) for id in 1:k]
        #Covariances
        for id in 1:k
            push!(covariances,cov(@views(X[identities.==id,:]),corrected=false))
        end
        #Weights
        weights = [sum(identities.==id)/nCells for id in 1:k]
    else
        error("initialization must be 'kmeans', 'random' or a vector of specifying the identities of the samples.")
    end

    return centers, covariances, weights, identities
end

function initializationGaussianMixtureHyperparameters(X,μ0,Σ0)
    
    if μ0 === nothing
        μ0 = reshape(mean(X,dims=1),size(X)[2])
    elseif size(μ0) != (size(X)[2],)
        error("μ0 must be or nothing or vector of the same size as dimensions in the system.")
    end

    if Σ0 === nothing
        Σ0 = cov(X,corrected=false)
    elseif size(Σ0) != (size(X)[2],size(X)[2])
        error("Σ0 must be or nothing or a squared matrix of the same size as dimensions in the system.")
    end

    return μ0,Σ0
end

"""
function gaussianMixtureEM(fct::Matrix;
    k::Int,
    initialization::Union{String,Matrix} = "kmeans",
    maximumSteps::Int = 10000                                
    )

Clustering with multivariate gaussian distributions using Expectation Maximization point estimates.

**Arguments**
- **fct::Matrix** Matrix to cluster the data with shape (samples,dimensions).

**Keyword Arguments**
- **k::Int** Number of clusters of the model.
- **initialization::Union{String,Matrix} = "kmeans"** Initialization. Select between "kmeans", "random" or give a matrix of (k,dimensions)
- **maximumSteps::Int = 10000** Number of maximum of steps before stopping the algorithm.

**Returns**
####################TO BE PUT
"""
function finiteGaussianMixtureEM(X::Matrix;
                            k::Int,
                            initialization::Union{String,Vector} = "kmeans",
                            maximumSteps::Int = 10000                                
                            )

    nCells, dimensions = size(X)

    #Initialization
    centers,covariances,weights,identities = initializationGaussianMixture(X,k,initialization)

    #Loop
    p = zeros(nCells,k)
    steps = 0
    identitiesNew = fill(0,nCells)
    while !all(identities.==identitiesNew) && steps < maximumSteps
        identities .= identitiesNew
        #Maximization step
        gmloglikelihood!(p,X,centers,covariances,weights)
        identitiesNew .= Matrix([i[2] for i in argmax(p,dims=2)])[:,1]
        #Expectation step
        for i in 1:k
            ids = identitiesNew.==i

            weights[i] = sum(ids)/nCells
            if weights[i] > 0            
                centers[i] .= mean(X[ids,:],dims=1)[1,:]
                covariances[i] .= cov(X[ids,:],corrected=false)
            end
        end
        steps += 1
    end

    #Fitted distribution
    dist = MixtureModel(MultivariateNormal[[MultivariateNormal(i,j) for (i,j) in zip(centers,covariances)]...],weights)

    return dist, identities

end

function gmIdentitiesProbability!(p::Matrix,X::Matrix,centers::Matrix,covariances::Array,weights::Vector)

    gmloglikelihood!(p,X,centers,covariances,weights)

    #Normalize taking the maximum
    pmax = maximum(p,dims=2)
    p .-= pmax

    #Exp
    p .= exp.(p)

    #Normalize
    p ./= sum(p,dims=2)

    return
end

"""
    finiteGaussianMixture(X::Matrix;
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
        verbose = false
        )

Function to fit a finite mixture model to data.

Arguments:

 - **X::Matrix**: Matrix with data to be fitted as (realizations, dimensions).

Keyword Arguments:

 - **k::Int**: Number of components of the mixture.
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

A GaussianFiniteMixtureModel with the sampling of the bayesian model.
"""
function finiteGaussianMixture(X::Matrix;
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

    Random.seed!(seed)

    nCells, dimensions = size(X)

    #Activate progress line
    if verbose 
        ProgressMeter.ijulia_behavior(:clear)
        prog = Progress(length(ignoreSteps+saveSteps))
    end 

    #Initialization
    centers,covariances,weights,identities = initializationGaussianMixture(X,k,initialization)
    μ0,Σ0 = initializationGaussianMixtureHyperparameters(X,μ0,Σ0)

    #Auxiliar functions
    p = zeros(nCells,k)
    votes = fill(0,k) #Auxiliar for sampling from the Dirichlet distributions
    vote = fill(0,k) #Auxiliar for sampling from the Dirichlet distributions

    saveIds = Vector{Int}[]
    saveDist = MixtureModel[]

    #Loop
    for step in 1:(ignoreSteps+saveSteps)
        #Sample identities
        gmlikelihood!(p,X,centers,covariances,weights)
        votes .= 0 #Reset votes
        for i in 1:nCells           
            vote .= rand(Multinomial(1,@views(p[i,:])))
            votes .+= vote
            identities[i] = findfirst(vote.==1)
        end
        #Sample parameters
            #Sample weights
        weights .= rand(Dirichlet(votes.+α/k))
        for comp in 1:k
            ids = identities.==comp

            if votes[comp] > dimensions #Check if we have enough statistical power to compute the wishart            
                #Statistics
                m = reshape(mean(X[ids,:],dims=1),dimensions)
                S2 = cov(@views(X[ids,:]),corrected=false)
                #Sample covariance
                neff = votes[comp]+ν0+1

                Σeff = votes[comp]*S2 + votes[comp]*(centers[comp]-m)*transpose((centers[comp]-m)) + κ0*(centers[comp]-μ0)*transpose((centers[comp]-μ0)) + Σ0
                Σeff = (Σeff+transpose(Σeff))/2 #Reinforce hemicity
                covariances[comp] = rand(InverseWishart(neff,Σeff))
                #Sample centers
                μeff = (votes[comp]*m+κ0*μ0)/(votes[comp]+κ0)
                centers[comp] .= rand(MultivariateNormal(μeff,covariances[comp]/(votes[comp]+κ0)))

            end
        end

        #Save
        if step >= ignoreSteps && step%saveEach == 0
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

    return GaussianFiniteMixtureModel(k,
                                Dict([  
                                        :α=>α,
                                        :ν0 => ν0,
                                        :κ0 => κ0,
                                        :μ0 => μ0,
                                        :Σ0 => Σ0
                                    ]),
                                saveDist,
                                saveIds)
end

"""
    infiniteGaussianMixture(X::Matrix;
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

Function to fit a infinite mixture model to data.

Arguments:

 - **X::Matrix**: Matrix with data to be fitted as (realizations, dimensions).

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

Return 

A GaussianInfiniteMixtureModel with the sampling of the bayesian model.
"""
function infiniteGaussianMixture(X::Matrix;
    k = 1,
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

    Random.seed!(seed)

    nCells, dimensions = size(X)

    #Activate progress line
    if verbose 
        ProgressMeter.ijulia_behavior(:clear)
        prog = Progress(length(ignoreSteps+saveSteps))
    end 

    #Initialization
    centers,covariances,weights,identities = initializationGaussianMixture(X,k,initialization)
    #Statistics
    m = deepcopy(centers)
    S2 = deepcopy(covariances)
    n = Int[sum(identities.==i) for i in 1:k]
    #Hyperparameters
    μ0,Σ0 = initializationGaussianMixtureHyperparameters(X,μ0,Σ0)
    #Effective parameters
    m0 = Float64[]
    m1 = Float64[]
    νeff = Float64[]
    μyeff = Vector{Float64}[]
    Σyeff = Matrix{Float64}[]
    for j in 1:k
        push!(m0,(n[j]+κ0)/(n[j]+κ0+1))
        push!(m1,((n[j]*κ0)/(n[j]+κ0)+n[j]*κ0)/(n[j]+κ0+1))
        push!(νeff,n[j]+ν0+1-dimensions)
        push!(μyeff,(n[j]*m[j]+κ0*μ0)/(n[j]+κ0))
        aux = (m1[j]*(μ0-m[j])*transpose(μ0-m[j])+n[j]*S2[j]+Σ0)/m0[j]
        push!(Σyeff,(aux+transpose(aux))/2)
    end
    #Auxiliar functions
    mnew = zeros(dimensions)
    w = zeros(k+1)
    vote = fill(0,k) #Auxiliar for sampling from the Dirichlet distributions

    saveIds = Vector{Int}[]
    saveDist = MixtureModel[]

    #Loop
    for step in 1:(ignoreSteps+saveSteps)

        for i in 1:nCells           

            id = identities[i]

            #Remove sample
            if n[id] == 1 #Remove component
                popat!(centers,id)
                popat!(covariances,id)
                popat!(weights,id)
                popat!(m,id)
                popat!(S2,id)
                popat!(n,id)
                popat!(w,id)
                popat!(m0,id)
                popat!(m1,id)
                popat!(νeff,id)
                popat!(μyeff,id)
                popat!(Σyeff,id)
                identities[identities.>id] .-= 1 #Reduce index of components above the removed one
                k -= 1
            else #Modify statistics
                #Statistics
                mnew .= (n[id]*m[id]-X[i,:])/(n[id]-1)
                S2[id] .= (n[id]*S2[id] + n[id]*m[id]*transpose(m[id]) - X[i,:]*transpose(X[i,:]))/(n[id]-1) - mnew*transpose(mnew)
                m[id] .= mnew
                n[id] -= 1 
                #Effective parameters
                m0[id] = (n[id]+κ0)/(n[id]+κ0+1)
                m1[id] = ((n[id]*κ0)/(n[id]+κ0)+n[id]*κ0)/(n[id]+κ0+1)
                νeff[id] = n[id]+ν0+1-dimensions
                μyeff[id] = (n[id]*m[id]+κ0*μ0)/(n[id]+κ0)
                aux = (m1[id]*(μ0-m[id])*transpose(μ0-m[id])+n[id]*S2[id]+Σ0)/m0[id]
                Σyeff[id] = (aux+transpose(aux))/2

                m0[id] = (n[id]+κ0)/(n[id]+κ0+1)
                m1[id] = ((n[id]*κ0)/(n[id]+κ0)+n[id]*κ0)/(n[id]+κ0+1)
                νeff[id] = n[id]+ν0+1-dimensions
                μyeff[id] = (n[id]*m[id]+κ0*μ0)/(n[id]+κ0)
                Σyeff[id] = (m1[id]*(μ0-m[id])*transpose(μ0-m[id])+n[id]*S2[id]+Σ0)/m0[id]
                Σyeff[id] = (Σyeff[id]+transpose(Σyeff[id]))/2 #Solve problem with hermitian
            end

            #Reassign sample
            w .= 0
            for j in 1:k
                try
                    w[j] = logpdf(MvTDist(νeff[j],μyeff[j],Σyeff[j]/νeff[j]),@views(X[i,:]))+log(n[j]/(nCells+α-1)) 
                catch
                    println("Asignation failed.")
                    error(Σyeff[j]/νeff[j])
                end
            end
            w[end] = logpdf(MultivariateNormal(μ0,Σ0),@views(X[i,:]))+log(α/(nCells+α-1))
            w .-= maximum(w)
            w = exp.(w)
            w ./= sum(w)
            vote = rand(Multinomial(1,w))
            identities[i] = findfirst(vote.==1)

            #Update statistics
            id = identities[i]
            if id == k + 1 #Add component
                push!(m,X[i,:])
                push!(S2,zeros(dimensions,dimensions))
                push!(n,1)
                push!(centers,copy(μ0))
                push!(covariances,copy(Σ0))
                push!(weights,1/nCells)
                push!(w,0.)
                push!(m0,(n[id]+κ0)/(n[id]+κ0+1))
                push!(m1,((n[id]*κ0)/(n[id]+κ0)+n[id]*κ0)/(n[id]+κ0+1))
                push!(νeff,n[id]+ν0+1-dimensions)
                push!(μyeff,(n[id]*m[id]+κ0*μ0)/(n[id]+κ0))
                aux = (m1[id]*(μ0-m[id])*transpose(μ0-m[id])+n[id]*S2[id]+Σ0)/m0[id]
                push!(Σyeff,(aux+transpose(aux))/2)

                k += 1
            else #Modify statistics
                #Statistics
                mnew .= (n[id]*m[id]+X[i,:])/(n[id]+1)
                S2[id] = (n[id]*S2[id] + n[id]*m[id]*transpose(m[id]) + X[i,:]*transpose(X[i,:]))/(n[id]+1) - mnew*transpose(mnew)
                m[id] .= mnew
                n[id] += 1
                #Effective parameters
                m0[id] = (n[id]+κ0)/(n[id]+κ0+1)
                m1[id] = ((n[id]*κ0)/(n[id]+κ0)+n[id]*κ0)/(n[id]+κ0+1)
                νeff[id] = n[id]+ν0+1-dimensions
                μyeff[id] = (n[id]*m[id]+κ0*μ0)/(n[id]+κ0)
                Σyeff[id] = (m1[id]*(μ0-m[id])*transpose(μ0-m[id])+n[id]*S2[id]+Σ0)/m0[id]
                Σyeff[id] = (Σyeff[id]+transpose(Σyeff[id]))/2 #Solve problem with hermitian
            end

        end

        #Save
        if step >= ignoreSteps && step%saveEach == 0

            weights .= rand(Dirichlet(n.+α/k))
            for comp in 1:k

                if n[comp] > dimensions #Check if we have enough statistical power to compute the wishart            
                    #Sample covariance
                    neff = n[comp]+ν0+1

                    Σeff = n[comp]*S2[comp] + κ0*n[comp]/(κ0+n[comp])*(m[comp]-μ0)*transpose(m[comp]-μ0)+Σ0
                    Σeff = (Σeff+transpose(Σeff))/2 #Reinforce hemicity
                    covariances[comp] = rand(InverseWishart(neff,Σeff))
                    #Sample centers
                    μeff = (n[comp]*m[comp]+κ0*μ0)/(n[comp]+κ0)
                    centers[comp] .= rand(MultivariateNormal(μeff,covariances[comp]/(n[comp]+κ0)))
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

    return GaussianInfiniteMixtureModel(
                                Dict([  
                                        :α=>α,
                                        :ν0 => ν0,
                                        :κ0 => κ0,
                                        :μ0 => μ0,
                                        :Σ0 => Σ0
                                    ]),
                                saveDist,
                                saveIds)

end


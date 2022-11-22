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

function gmloglikelihood!(p::Matrix,X::Matrix,centers::Matrix,covariances::Array,weights::Vector)

    for j in 1:size(centers)[1]
        c = @views centers[j,:]
        sigma = covariances[j,:,:]
        sigmaInv = inv(sigma)
        determinant = abs(det(sigma))
        w = weights[j]
        for i in 1:size(X)[1]
            x = @views X[i,:]
            m = (x-c).^2
            p[i,j] = -(transpose(m)*sigmaInv*m/2)[1]-log(determinant)/2+log(w)
        end
    end

    return
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
                            initialization::Union{String,Matrix} = "kmeans",
                            maximumSteps::Int = 10000                                
                            )

    #Check shape of initial centers
    if typeof(initialization) == Matrix
        if size(initialisation) != (k,size(X)[2])
            error("If initialization is a matrix of given initial positions, the dimensions must be the same size as (k,variables).")
        end
    end

    nCells = size(X)[1]

    #Initialization centers
    centers = zeros(k,size(X)[2])
    if initialization == "kmeans"
        model = kmeans_(k=k)
        mach = machine(model,X,scitype_check_level=0)
        fit!(mach,verbosity=0)
        centers = permutedims(fitted_params(mach)[1])
    elseif initialization == "random"
        centers = rand(k,size(X)[2])
        centers .= centers .* (maximum(X,dims=1).-minimum(X,dims=1)) .-minimum(X,dims=1)
    elseif typeof(initialization) == Matrix
        centers .= initialization
    else
        error("initialization must be 'kmeans', 'random' or a matrix of specifying the centers of the gaussian.")
    end
    #Initialization of identities
    identities = closerPoints(X,centers)
    #Initialization of covariances
    covariances = []
    for id in 1:k
        push!(covariances,cov(@views X[identities.==id,:]))
    end
    #Initialization of weights
    weights = [sum(identities.==id)/nCells for id in 1:k]

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
                centers[i,:] .= mean(X[ids,:],dims=1)[1,:]
                covariances[i] .= cov(X[ids,:])
            end
        end
        steps += 1
    end

    fct.uns[key_added] = Dict([
        "k" => k,
        "maximumSteps" => maximumSteps,
        "stepsBeforeConvergence" => steps,
        "initialization" => initialization,
        "weights" => p
    ])

    fct.obs[!,key_added] = identitiesNew

    return

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

function finiteGaussianMixture(fct::Matrix;
    k::Int,
    initialization::Union{String,Matrix} = "kmeans",
    ignoreSteps::Int = 1000, 
    saveSteps::Int = 1000,
    saveEach::Int = 10,
    key_added::String = "gaussianMixture",
    key_obsm::Union{Nothing,String} = nothing,
    n_components::Union{Nothing,Int} = nothing,
    key_used_channels::Union{Nothing,String} = nothing,
    verbose = false
    )

    ProgressMeter.ijulia_behavior(:clear)

    if key_obsm !== nothing && key_used_channels !== nothing
    error("key_obsm and key_used_channels cannot be specified at the same time.")
    elseif key_obsm !== nothing
    if n_components !== nothing
    X = fct.obsm[key_obsm][:,1:n_components]
    else
    X = fct.obsm[key_obsm]
    end
    else
    if key_used_channels !== nothing
    channels = fct.var[:,key_used_channels]
    if typeof(channels) != Vector{Bool}
    error("key_used_channels should be a column in var of Bool entries specifying which channels to use for clustering.")
    end

    X = fct.X[:,channels]
    else
    X = fct.X
    end
    end

    #Check shape of initial centers
    if typeof(initialization) == Matrix
    if size(initialisation) != (k,size(X)[2])
    error("If initialization is a matrix of given initial positions, the dimensions must be the same size as (k,variables).")
    end
    end

    nCells = size(X)[1]
    dims = size(X)[2]

    #Initialization centers
    centers = zeros(k,dims)
    if initialization == "kmeans"
    model = kmeans_(k=k)
    mach = machine(model,X,scitype_check_level=0)
    fit!(mach,verbosity=0)
    centers = permutedims(fitted_params(mach)[1])
    elseif initialization == "random"
    centers = rand(k,dims)
    centers .= centers .* (maximum(X,dims=1).-minimum(X,dims=1)) .-minimum(X,dims=1)
    elseif typeof(initialization) == Matrix
    centers .= initialization
    else
    error("initialization must be 'kmeans', 'random' or a matrix of specifying the centers of the gaussian.")
    end
    #Initialization of identities
    identities = closerPoints(X,centers)
    identitiesRef = copy(identities)
    #Initialization of covariances
    covariances = zeros(k,dims,dims)
    for id in 1:k
    votes = identities.==id
    if sum(votes) > dims
    covariances[id,:,:] .= cov(@views X[votes,:])
    else
    for i in 1:dims
    covariances[id,i,i] = 1
    end
    end
    end
    #Initialization of weights
    weights = [sum(identities.==id)/nCells for id in 1:k]

    #Loop
    p = zeros(nCells,k)
    steps = 0
    votes = fill(0,k) #Auxiliar for sampling from the Dirichlet distributions
    vote = fill(0,k) #Auxiliar for sampling from the Dirichlet distributions
    nSave = sum((ignoreSteps:(ignoreSteps+saveSteps)).%saveEach .== 0)
    saveMeans = zeros(nSave,k,dims)
    saveCovariances = zeros(nSave,k,dims,dims)
    saveWeights = zeros(nSave,k)
    countSave = 0
    itrs = 1:(ignoreSteps+saveSteps)
    if verbose 
    #itrs = ProgressBar(1:(ignoreSteps+saveSteps))
    prog = Progress(length(itrs))
    end 
    for step in itrs
    #Sample identities
    gmIdentitiesProbability!(p,X,centers,covariances,weights)
    for i in 1:nCells           
    vote .= rand(Multinomial(1,p[i,:]))
    votes .+= vote
    identities[i] = findfirst(vote.==1)
    end
    #Sample parameters
    #Sample weights
    weights .= rand(Dirichlet(votes.+1))
    for i in 1:k
    ids = identities.==i

    if votes[i] > dims #Check if we have enough statistical power to compute the wishart            
    # Sample covariances
    c = cov(X[ids,:])
    w = rand(InverseWishart(votes[i],votes[i].*c))
    covariances[i,:,:] .= w
    m = reshape(mean(X[ids,:],dims=1),dims)
    centers[i,:] .= rand(MultivariateNormal(m,w))
    end
    end

    if step >= ignoreSteps && step%saveEach == 0
    rel = relabeling(identities,identitiesRef)
    countSave += 1
    saveWeights[countSave,:] .= copy(weights)[rel]
    saveCovariances[countSave,:,:,:] .= copy(covariances)[rel,:,:]
    saveMeans[countSave,:,:] .= copy(centers)[rel,:]
    end

    if verbose
    next!(prog,showvalues=[(:iter,step)])
    end

    votes .= 0
    end

    fct.uns[key_added] = Dict([
    "k" => k,
    "initialization" => initialization,
    "ignoreSteps" => ignoreSteps, 
    "saveSteps" => saveSteps,
    "saveEach" => saveEach,
    "weights" => saveWeights,
    "covariances" => saveCovariances,
    "means" => saveMeans,
    ])

    return
end

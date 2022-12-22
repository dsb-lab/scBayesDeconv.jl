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

function finiteGaussianMixtureDeconvolution(X::Matrix, Y::GaussianFiniteMixtureModel;
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
        for comp in 1:k
            idsT = identities.==comp

            if votesK[comp] > dimensions #Check if we have enough statistical power to compute the wishart  

                #Sample covariance
                m2 .= 0
                S2 .= 0   
                for compN in 1:kN
                    idsN = identitiesN.==compN
                    ids = idsN .& idsT
                    #Statistics
                    aux = (reshape(mean(X[ids,:],dims=1),dimensions)-centersN[compN]-centers[comp])
                    m2 .+= votes[cartesian2lin(comp,compN,k,kN)]*aux*transpose(aux)
                    S2 .+= votes[cartesian2lin(comp,compN,k,kN)]*(cov(@views(X[ids,:]),corrected=false)-covariancesN[compN])
                end
                # println(votes,m/votesK[comp],S2/votesK[comp])
                for i in 1:dimensions
                    if S2[i,i] < 0
                        S2[i,i] = 0
                    end
                end
                neff = votesK[comp]+ν0+1

                Σeff = S2 + m2 + κ0*(centers[comp]-μ0)*transpose((centers[comp]-μ0)) + Σ0
                Σeff = (Σeff+transpose(Σeff))/2 #Reinforce hemicity
                covariances[comp] = rand(InverseWishart(neff,Σeff))
                #Sample centers
                m .= 0
                S2 .= 0   
                for compN in 1:kN
                    idsN = identitiesN.==compN
                    ids = idsN .& idsT
                    s = inv(covariances[comp]+covariancesN[compN])
                    #Statistics
                    # println((votes[cartesian2lin(comp,compN,k,kN)]*(reshape(mean(X[ids,:],dims=1),dimensions)-centersN[compN])+κ0*μ0))
                    m .+= s*(votes[cartesian2lin(comp,compN,k,kN)]*(reshape(mean(X[ids,:],dims=1),dimensions)-centersN[compN])+κ0*μ0)
                    S2 .+= s*(votes[cartesian2lin(comp,compN,k,kN)]+κ0)
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
            push!(saveN)
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

function infiniteGaussianMixtureDeconvolution(X::Matrix, Y::GaussianInfiniteMixtureModel;
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

    #Loop
    for step in 1:(ignoreSteps+saveSteps)

        #Sample noise distribution
        nSample = rand(nRange)
        centersN = [i.μ for i in Y.samples[nSample].components]
        covariancesN = [i.Σ for i in Y.samples[nSample].components]
        weightsN = Y.samples[nSample].prior.p        
        kT = length(centers)
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
            for j in 1:kN
                for k in 1:kT
                    try
                        w[cartesian2lin(j,k,kN,kT+1)] = logpdf(MvTDist(νeff[j,k],μyeff[j,k],Σyeff[j,k]/νeff[j,k]),@views(X[i,:]))+log(n[j,k]/(nCells+α-1))
                    catch
                        error((j,k,cartesian2lin(j,k,kN,kT+1)),"\n S2",S2,"\n Σyeff",Σyeff,"\n m1",m1,"\n m0",m0,"\n")
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
                    m2 = zeros(dimensions,dimensions)
                    S2 = zeros(dimensions,dimensions)
                    for compN in 1:kN
                        idsN = identitiesN.==compN
                        ids = idsN .& idsT
                        #Statistics
                        if sum(ids) > 0
                            aux = (reshape(mean(X[ids,:],dims=1),dimensions)-centersN[compN]-centers[comp])
                            m2 .+= n[compN,comp]*aux*transpose(aux)
                            S2 .+= n[compN,comp]*(cov(@views(X[ids,:]),corrected=false)-covariancesN[compN])
                        end
                    end
                    # println(votes,m/votesK[comp],S2/votesK[comp])
                    for i in 1:dimensions
                        if S2[i,i] < 0
                            S2[i,i] = 0
                        end
                    end
                    neff = votesK[comp]+ν0+1
    
                    Σeff = S2 + m2 + κ0*(centers[comp]-μ0)*transpose((centers[comp]-μ0)) + Σ0
                    # println(S2, m2, κ0*(centers[comp]-μ0)*transpose((centers[comp]-μ0)), Σ0)
                    Σeff = (Σeff+transpose(Σeff))/2 #Reinforce hemicity
                    covariances[comp] = rand(InverseWishart(neff,Σeff))
                    #Sample centers
                    m = zeros(dimensions)
                    S2 = zeros(dimensions,dimensions)
                    for compN in 1:kN
                        idsN = identitiesN.==compN
                        ids = idsN .& idsT
                        s = inv(covariances[comp]+covariancesN[compN])
                        #Statistics
                        # println((votes[cartesian2lin(comp,compN,k,kN)]*(reshape(mean(X[ids,:],dims=1),dimensions)-centersN[compN])+κ0*μ0))
                        if sum(ids) > 0
                            m .+= s*(n[compN,comp]*(reshape(mean(X[ids,:],dims=1),dimensions)-centersN[compN])+κ0*μ0)
                            S2 .+= s*(n[compN,comp]+κ0)
                        end
                    end
                    S2 = inv(S2)
                    m = S2*m
                    centers[comp] .= rand(MultivariateNormal(m,S2))
    
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
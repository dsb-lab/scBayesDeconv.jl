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

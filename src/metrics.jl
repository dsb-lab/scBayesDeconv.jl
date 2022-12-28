module metrics

    import ..scBayesDeconv: GaussianMixtureModel
    import HypothesisTests: ExactOneSampleKSTest

    function decompose_(n,s)

        cp = cumprod(s)[1:end-1]
        cp = cp[end:-1:1]
        push!(cp,1)
        l = length(s)
        c = zeros(l) 
        for i in 1:l
            c[l-i+1] = n÷cp[i]
            n -= c[l-i+1]*cp[i]
        end 

        return c
    end

    function MISE(model::GaussianMixtureModel,f2::Function;box=[-100. 100.],d=.1,samples=1:length(f1.samples))

        mises = Float64[]
        for i in samples
            f1(x) = pdf(model.samples[i],x)
            push!(mises,MISE(ft,f2,box=box,d=d))
        end
    
        return mios
    
    end

    function MISE(f1::Function,f2::Function;box,d)

        mise = 0
        dd = d^size(box)[1]
        dx = (box[:,2]-box[:,1])/d
        s = Int.(round.(dx))
        for i in 0:prod(s)
            point = decompose_(i,s).*d .+box[:,1]
            mise += (f1(point)-f2(point))^2*dd/2
        end
    
        return mise
    
    end

    function MIO(model::GaussianMixtureModel,f2::Function;box=[-100. 100.],d=.1,samples=1:length(f1.samples))

        mios = Float64[]
        for i in samples
            f1(x) = pdf(model.samples[i],x)
            push!(mios,MIO(ft,f2,box=box,d=d))
        end
    
        return mios
    
    end

    function MIO(f1::Function,f2::Function;box=box,d=.1)

        mio = 1
        dd = d^size(box)[1]
        dx = (box[:,2]-box[:,1])/d
        s = Int.(round.(dx))
        for i in 0:prod(s)
            point = decompose_(i,s).*d .+box[:,1]
            mio -= abs(f1(point)-f2(point))*dd/2
        end
    
        return mio
    
    end

    function KolmogorovSmirnov(f1::GaussianMixtureModel,f2::Function;nSamples=1000)

        ks = Float64[]
        for i in samples
            xx = rand(model.samples[i],nSamples)
            push!(mios,ExactOneSampleKSTest(xx,f2))
        end
    
        return ks
    
    end


end
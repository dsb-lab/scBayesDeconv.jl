module metrics

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

    function MISE(f1,f2,box,d)

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

    function MIAE(f1,f2,box,d)

        miae = 1
        dd = d^size(box)[1]
        dx = (box[:,2]-box[:,1])/d
        s = Int.(round.(dx))
        for i in 0:prod(s)
            point = decompose_(i,s).*d .+box[:,1]
            miae -= abs(f1(point)-f2(point))*dd/2
        end
    
        return miae
    
    end

end
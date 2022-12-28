function newmannDeconvolutionNaturalEstimate_(data,w)

    sol = zeros(Complex,length(w))
    l = size(data)[1]
    for (n,i) in enumerate(w)
        sol[n] = sum(exp.(1im*i*data)/l)
    end

    return sol

end

function newmannDeconvolution(aut,data,d1=1,d2=1,dw=0.01,w_lims=[-100,100],dx=0.01,x_lims=[-100,100],cut_off=true)

    w = range(w_lims[1],w_lims[2],step=dw)
    x = range(x_lims[1],x_lims[2],step=dx)
    hn = sqrt((length(aut)+1)^(1/(2*d1+2*d2))-1)
    dist = Normal(0,hn)
    Kw = pdf.(dist,w)
    phiY = newmannDeconvolutionNaturalEstimate_(data,w)
    phie = newmannDeconvolutionNaturalEstimate_(aut,w)

    if cut_off
        #Generate the cutoff
        I = ones(length(Kw))
        I[abs.(phie) .<= length(aut)^(-.5)].=0
        a1 = I .*phiY .*Kw ./phie
        a2 = exp.(-1im .* reshape(w,1,length(w)) .*x)
        deconv = (a2 * a1) .*dw
    else
        deconv = (phiY .*Kw ./phie * exp.(-1im .*reshape(w,length(w),1) .*x)) .*dw
    end

    return real.(deconv), x

end
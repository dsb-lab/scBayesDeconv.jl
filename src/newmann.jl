function newmannDeconvolutionNaturalEstimate_(data,w)

    sol = zeros(length(w))
    l = size(data)[0]
    for (n,i) in enumerate(w)
        sol[n] = sum(exp(1im*i*data)/l)
    end

    return sol

end

function newmannDeconvolution(aut,data,dw=0.01,w_lims=[-100,100],dx=0.1,x_lims=[-100,100],cut_off=true)

    w = range(w_lims[0],w_lims[1],step=dw)
    x = range(x_lims[0],x_lims[1],step=dx)
    Kw = st.norm.pdf(w,0,0.1)
    phiY = newmannDeconvolutionNaturalEstimate_(data,w)
    phie = newmannDeconvolutionNaturalEstimate_(aut,w)

    if cut_off
        #Generate the cutoff
        I = ones(length(Kw))
        I[phie .<= length(aut)^(-.5)]=0
        deconv = (I*phiY*Kw/phie,exp(-1im*w.reshape(-1,1)*x))*dw
    else
        deconv = (phiY*Kw/phie,exp(-1im*w.reshape(-1,1)*x))*dw
    end

    return deconv, x

end
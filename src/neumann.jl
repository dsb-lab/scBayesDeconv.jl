struct Neumann 
    interpolate
end

(y::Neumann)(x::Matrix) = y.interpolate.(x[:,1])
(y::Neumann)(x::Vector) = y.interpolate.(x)
(y::Neumann)(x::Real) = y.interpolate(x)

function neumannDeconvolutionNaturalEstimate_(data,w)

    sol = zeros(Complex,length(w))
    l = size(data)[1]
    for (n,i) in enumerate(w)
        sol[n] = sum(exp.(1im*i*data)/l)
    end

    return sol

end

"""
    function neumannDeconvolution(noise::Matrix,conv::Matrix,d1=1,d2=1,dw=0.01,w_lims=[-100,100],dx=0.01,x_lims=[-100,100],cut_off=true)

Fast-Fourier method for the deconvolution of two distributions as proposed by (Neumann & Hössjer)[https://www.tandfonline.com/doi/abs/10.1080/10485259708832708]. 
This method as implemented only works for 1D distributions.

Arguments:

 - **noise::Matrix**: Autofluorescence data of size (NSamples,1).
 - **conv::Matrix**: Convolution data of size (NSamples,1).
 - **d1=1**: Weight of the estimation of the frequencies to be removed based on the size of the data.
 - **d2=1**: Weight of the estimation of the frequencies to be removed  based on the size of the data.
 - **dw=0.01**: Width of the frequency sampling.
 - **w_lims=[-100,100]**: Limits of the frequency domain.
 - **dx=0.01**: Width of the spatial sampling.
 - **x_lims=[-100,100]**: Limits of the spatial domain.
 - **cut_off=true**: If to use the cutoff of the frequencies (in the original method this is true).

Returns:

Deconvolved signal, Points of the deconvolved signal determined by **x_lims** and **dx**.
"""
function neumannDeconvolution(noise::Matrix,conv::Matrix,d1=1,d2=1,dw=0.01,w_lims=[-100,100],dx=0.01,x_lims=[-100,100],cut_off=true)

    aut = noise[:,1]
    data = conv[:,1]

    w = range(w_lims[1],w_lims[2],step=dw)
    x = range(x_lims[1],x_lims[2],step=dx)
    hn = sqrt((length(aut)+1)^(1/(2*d1+2*d2))-1)
    dist = Normal(0,hn)
    Kw = pdf.(dist,w)
    phiY = neumannDeconvolutionNaturalEstimate_(data,w)
    phie = neumannDeconvolutionNaturalEstimate_(aut,w)

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

    return Neumann(Interpolations.LinearInterpolation(x,real.(deconv),extrapolation_bc=0))

end
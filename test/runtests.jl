using Test
using scBayesDeconv
using Distributions
using Random

d = MixtureModel(
    MultivariateNormal[
        MultivariateNormal([0; 1],[1 .5;.5 1])
    ],
    [1.]
    )

d = MixtureModel(
    MultivariateNormal[
        MultivariateNormal([1],2*ones(1,1))
    ],
    [1.]
    )

X = Matrix(transpose(rand(d,100)));

Y = Matrix(transpose(rand(d,100)))#.+Matrix(transpose(rand(d,100)));


@testset "Test" begin
    
    # @test_nowarn scBayesDeconv.initializationGaussianMixture(X,2,"kmeans")
    # @test_nowarn scBayesDeconv.initializationGaussianMixture(X,2,"random")
    # @test_nowarn scBayesDeconv.initializationGaussianMixture(X,2,rand([1,2],size(X)[1]))

    # @test_nowarn dist,ids = finiteGaussianMixtureEM(X,k=1)

    # @test_nowarn begin
    #     x = finiteGaussianMixture(X,k=1,verbose=true)
    #     println(x.samples[1])
    # end

    # @test_nowarn x = infiniteGaussianMixture(X,k=1,verbose=true)

    # @test_nowarn begin 
    #     x = finiteGaussianMixture(X,k=1,verbose=true)
    #     y = finiteGaussianMixtureDeconvolution(Y,x,k=1,verbose=true)
    # end

    @test_nowarn begin 
        x = infiniteGaussianMixture(X,k=1,verbose=true)
        y = infiniteGaussianMixtureDeconvolution(Y,x,k=1,verbose=true)
    end

end
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

X = Matrix(transpose(rand(d,100)));

@testset "Test" begin
    
    # @test_nowarn scBayesDeconv.initializationGaussianMixture(X,2,"kmeans")
    # @test_nowarn scBayesDeconv.initializationGaussianMixture(X,2,"random")
    # @test_nowarn scBayesDeconv.initializationGaussianMixture(X,2,rand([1,2],size(X)[1]))

    # @test_nowarn dist,identities=finiteGaussianMixtureEM(X,k=1)

    # @test_nowarn dist,identities = finiteGaussianMixture(X,k=1,verbose=true)

    

end
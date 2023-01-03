
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gatocor.github.io/scBayesDeconv.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gatocor.github.io/scBayesDeconv.jl/dev/)
# scBayesDeconv.jl

This is a package that implements the bayesian deconvolution methods for solving the following problem:

 ``C = T + \xi``

where ``C``, ``T`` and ``\xi`` are random variables and we have sample sets from ``C`` and ``\xi``; and we would like to know the distribution of the random variable ``T``.
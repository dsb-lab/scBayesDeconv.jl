
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gatocor.github.io/scBayesDeconv.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gatocor.github.io/scBayesDeconv.jl/dev/)
# scBayesDeconv.jl

This is a package that implements the bayesian deconvolution methods for solving the following problem:

$$C = T + \xi$$

where $C$ (convolution), $T$ (target) and $\xi$ (noise) are random variables and we have sample sets from $C$ and $\xi$; and we would like to know the distribution of the random variable $T$.

![svg](assets/Artificial%20Convolutions_21_0.svg)
## What is implemented in the package?

Bayesian Gaussian Mixture Models:

 - Finite Gaussian Mixture Models
 - Infinite Gaussian Mixture Models (Dirichlet Processes)

Deconvolution Bayesian Gaussian Mixture Models:

 - Finite Deconvolution Gaussian Mixture Models
 - Infinite Deconvolution Gaussian Mixture Models (Dirichlet Processes)
## Installation

The package can be installed ass

```
pkg> add github.com/gatocor/scBayesDeconv.jl#VERSION
```

or 

```
using Pkg
Pkg.add("github.com/gatocor/scBayesDeconv.jl#VERSION")
```
for the version of interest.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gatocor.github.io/scBayesDeconv.jl](https://github.com/dsb-lab/scBayesDeconv.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gatocor.github.io/scBayesDeconv.jl](https://github.com/dsb-lab/scBayesDeconv.jl/dev/)
# scBayesDeconv.jl

This package implements a Bayesian deconvolution method for extracting the distribution of a target signal $T$ from a measured signal $C$ subject to noise $\xi$:

$$C = T + \xi$$

In other words, we have sample sets from $C$ and $\xi$; and we would like to know the distribution of the signal $T$.

![svg](assets/Artificial%20Convolutions_21_0.svg)
## What is implemented in the package?

Bayesian Gaussian Mixture Models:

 - Finite Gaussian Mixture Models
 - Infinite Gaussian Mixture Models (Dirichlet Processes)

Deconvolution Bayesian Gaussian Mixture Models:

 - Finite Deconvolution Gaussian Mixture Models
 - Infinite Deconvolution Gaussian Mixture Models (Dirichlet Processes)
## Installation

The package can be installed as

```
pkg> add https://github.com/dsb-lab/scBayesDeconv.jl#VERSION
```

or 

```
using Pkg
Pkg.add("https://github.com/dsb-lab/scBayesDeconv.jl#VERSION")
```
for the version of interest.

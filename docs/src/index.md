# scBayesDeconv.jl

This is a package that implements the bayesian deconvolution methods for solving the following problem:

$$C = T + \xi$$
where $C$, $T$ and $\xi$ are random variables and we have sample sets from $C$ and $\xi$; and we would like to know the distribution of the random variable $T$.

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
pkg> add https://github.com/gatocor/scBayesDeconv.jl#VERSION
```

or 

```
using Pkg
Pkg.add("https://github.com/gatocor/scBayesDeconv.jl#VERSION")
```
for the version that you want to install.

## Getting started

To understand the basic usage of these models, it is advised to see the example `Artificial convolutions`.

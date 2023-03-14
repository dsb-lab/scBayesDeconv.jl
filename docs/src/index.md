# scBayesDeconv: A Julia package for noise deconvolution

This package implements a Bayesian deconvolution method for extracting the distribution of a target signal $T$ from a measured signal $C$ subject to noise $\xi$:

$$C = T + \xi$$

In other words, we have sample sets from $C$ and $\xi$; and we would like to know the distribution of the signal $T$.

## What is implemented in the package?

Bayesian Gaussian Mixture Models:

 - Finite Gaussian Mixture Models
 - Infinite Gaussian Mixture Models (Dirichlet Processes)

Deconvolution Bayesian Gaussian Mixture Models:

 - Finite Deconvolution Gaussian Mixture Models
 - Infinite Deconvolution Gaussian Mixture Models (Dirichlet Processes)
 - 
## Installation

The package can be installed in Julia as follows:

```
using Pkg
Pkg.add("https://github.com/dsb-lab/scBayesDeconv.jl#VERSION")
```
or alternatively

```
pkg> add https://github.com/dsb-lab/scBayesDeconv.jl#VERSION
```

for the version of interest (if no version is given, the development version will be installed).

## Getting started

To explore the basic usage of these models, see the example `Artificial convolutions`.

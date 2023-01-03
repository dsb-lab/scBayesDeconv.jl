# API

## Bayesian models

```@docs
finiteGaussianMixture
infiniteGaussianMixture
```

```@docs
finiteGaussianMixtureDeconvolution
infiniteGaussianMixtureDeconvolution
```
## Other deconvolution models

```@docs
neumannDeconvolution
```
## Metrics

### MISE

Is the Mean Integrated Squared Error metric defined as

``
MISE = \int (f_2(\bm{x})-f_1(\bm{x}))^2 d\bm{x}
``

```@docs
scBayesDeconv.metrics.mise
```

### MISE

Is the Mean Integrated Overlap metric defined as

``
MIO = 1 - \frac{1}{2}\int |f_2(\bm{x})-f_1(\bm{x})| d\bm{x}
``

which makes de metric bounded as ``MIO \in(0,1)``, being 1 is the distributions overlap perfectly and zero is there is no overlap in the domain at all.

```@docs
scBayesDeconv.metrics.mio
```
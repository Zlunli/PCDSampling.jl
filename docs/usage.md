# Usage Guide

This document describes the main user-facing API of `PCDSampling.jl`.

For runnable example scripts, see Running Example Code.

---

## 1. CPU Sampling with `draw_samples`

The main CPU entry point is `draw_samples`. It generates deterministic samples from a target multivariate distribution using a given set of projection directions.

There are two ways to call `draw_samples`.

---

### 1.1 Sampling Directly from a Target Distribution

Use this form when you have a target distribution and want the package to construct all projected target distributions automatically.

```julia
X, iters = draw_samples(dist, N, dirs;
    use_local=false,
    N_lut=-1,
    max_iters=100,
    eps=1e-6,
    init_samples=nothing,
    verbose=false,
    nthreads=Threads.nthreads(),
)
```

Example:

```julia
using LinearAlgebra
using Distributions
using PCDSampling

dist = MvNormal([0.0, 0.0], I(2))
dirs = uniform_directions_2d(200)

X, iters = draw_samples(dist, 1000, dirs; max_iters=100, N_lut=-1)
```

Important arguments:

- `dist`: target multivariate distribution.
  
- `N`: number of deterministic samples.
  
- `dirs`: projection directions.
  
- `N_lut`: lookup table size. Use `N_lut=-1` to disable lookup tables.
  
- `max_iters`: maximum number of optimization iterations.
  
- `eps`: stopping threshold for the sample update.
  
- `use_local`: if `true`, use the local update scheme; if `false`, use the global Newton-like update scheme.
  
- `init_samples`: optional initial sample positions. If omitted, initial samples are drawn randomly from `dist`.
  
- `verbose`: if `true`, print information about the final iteration and update norm.
  
- `nthreads`: number of CPU threads used by the algorithm. If omitted, the current number of Julia threads is used.
  

Returns:

- `X`: matrix containing the deterministic samples.
  
- `iters`: number of iterations performed.
  

---

### 1.2 Sampling from Precomputed Projections

Use this form when you have already created a `Projections` object and want to provide the initial samples yourself.

```julia
X, iters = draw_samples(projections, init_samples;
    max_iters=100,
    eps=1e-6,
    use_local=false,
    verbose=false,
    nthreads=Threads.nthreads(),
)
```

Example:

```julia
using LinearAlgebra
using Distributions
using PCDSampling

dist = MvNormal([0.0, 0.0], I(2))
dirs = reduce(hcat, uniform_directions_2d(200))

targets = [project(dist, d) for d in eachcol(dirs)]
projections = Projections(targets, dirs)

init_samples = rand(dist, 1000)

X, iters = draw_samples(projections, init_samples; max_iters=100)
```

This form is useful if you want to reuse the same projected target distributions or manually control how the projections are constructed.

---

## 2. GPU Sampling with `draw_samples_gpu`

The main GPU entry point is `draw_samples_gpu`. Its interface is similar to the CPU version, but the computation is performed on the GPU.

```julia
X, iters = draw_samples_gpu(dist, N, dirs;
    use_local=false,
    N_lut=128,
    max_iters=100,
    eps=1e-6,
    init_samples=nothing,
    verbose=false,
)
```

Example:

```julia
using CUDA
using LinearAlgebra
using Distributions
using PCDSampling

dist = MixtureModel([MvNormal([0.0, 0.0], I(2))])
dirs = uniform_directions_2d(200)

X, iters = draw_samples_gpu(dist, 1000, dirs; N_lut=128, max_iters=100)
```

Important arguments:

- `dist`: target multivariate distribution.
  
- `N`: number of deterministic samples.
  
- `dirs`: projection directions.
  
- `N_lut`: lookup table size. The GPU version requires lookup tables, so `N_lut` must be a positive integer.
  
- `max_iters`: maximum number of optimization iterations.
  
- `eps`: stopping threshold for the sample update.
  
- `use_local`: if `true`, use the local update scheme; if `false`, use the global Newton-like update scheme.
  
- `init_samples`: optional initial sample positions. If omitted, initial samples are drawn randomly from `dist`.
  
- `verbose`: if `true`, print information about the final iteration and update norm.
  

Returns:

- `X`: matrix containing the deterministic samples.
  
- `iters`: number of iterations performed.
  

---

## 3. Projection Directions

Both `draw_samples` and `draw_samples_gpu` require a set of projection directions. The package provides two helper functions for generating such directions.

### 3.1 Uniform Directions in 2D

Use `uniform_directions_2d` to generate uniformly spaced directions in the two-dimensional plane:

```julia
dirs = uniform_directions_2d(200)
```

This creates `200` two-dimensional unit direction vectors distributed uniformly over the interval `[0, π)`.

Here, `200` is the number of projection directions. Increasing this number provides more projection information, but also increases the computational cost.

### 3.2 Random Directions

Use `random_directions` to generate random unit direction vectors:

```julia
dirs = random_directions(dim, n_directions)
```

This creates `n_directions` random direction vectors in `dim` dimensions. Each vector is normalized to unit length.

Arguments:

- `dim`: dimension of the target distribution.
  
- `n_directions`: number of random projection directions.
  

The directions are generated by drawing Gaussian random vectors and normalizing them to unit length. This is based on the standard method by Muller (1959) for generating random directions on a sphere.

---

## 4. The `Projections` Structure

The `Projections` structure stores all one-dimensional projected target distributions together with their corresponding projection directions.

A `Projections` object contains:

- `projections`: a vector of one-dimensional projected target distributions.
  
- `dirs`: a direction matrix, where each column is one projection direction.
  

A `Projections` object can be constructed manually as follows:

```julia
using LinearAlgebra
using Distributions
using PCDSampling

dist = MvNormal([0.0, 0.0], I(2))
dirs = reduce(hcat, uniform_directions_2d(200))

targets = [project(dist, d) for d in eachcol(dirs)]
projections = Projections(targets, dirs)
```

Here, `project(dist, d)` computes the one-dimensional projected distribution of `dist` along direction `d`. The resulting `targets` vector contains one projected target distribution for each projection direction.

After constructing the `Projections` object, it can be passed directly to `draw_samples` together with initial sample positions:

```julia
init_samples = rand(dist, 1000)

X, iters = draw_samples(projections, init_samples; max_iters=100)
```

This form is useful when you want to reuse the same projections for multiple runs, for example with different initial samples, stopping criteria, or update schemes.

---

## 5. Stopping Conditions

The PCD sampling algorithm is iterative. It stops when the stopping condition returns `true`.

The package provides two helper functions for constructing stopping conditions:

- `fixed_iters`
  
- `max_iters_and_small_delta`
  

Users may also provide a custom stopping condition. A custom stopping condition should be a function that takes the current sample update as input and returns `true` when the algorithm should stop.

### 5.1 Fixed Number of Iterations

Use `fixed_iters` to run the algorithm for a fixed number of iterations:

```julia
stop_cond = fixed_iters(100)

X, iters = draw_samples(dist, 1000, dirs; stop_cond=stop_cond)
```

Here, `100` is the maximum number of iterations. This stopping condition only checks the iteration count and ignores the update error.

### 5.2 Maximum Iterations and Small Update

Use `max_iters_and_small_delta` to stop the algorithm based on both the iteration count and the update error:

```julia
stop_cond = max_iters_and_small_delta(100, 1e-6)

X, iters = draw_samples(dist, 1000, dirs; stop_cond=stop_cond)
```

Here:

- `100` is the maximum number of iterations.
  
- `1e-6` is the threshold for the update error.
  

This stopping condition stops the algorithm when either the maximum number of iterations is reached or the update error becomes sufficiently small.

The update error is measured using the infinity norm of the update matrix, i.e., the maximum absolute entry of the sample update.

---

## 6. Projecting Target Distributions

The function `project` computes the one-dimensional projected target distribution along a projection direction.

The relevant function signatures are:

```julia
project(dist::AbstractMvNormal, u)
project(dist::MultivariateMixture, u)
```

It is used to construct the projected target distributions required by the PCD sampling algorithm.

### 6.1 Multivariate Gaussian

For a multivariate Gaussian distribution, the projection is again a one-dimensional Gaussian.

```julia
using LinearAlgebra
using Distributions
using PCDSampling

dist = MvNormal([0.0, 0.0], I(2))
u = [1.0, 0.0]

proj_dist = project(dist, u)
```

If

```math
\mathbf{x} \sim \mathcal{N}(\mathbf{m}, \mathbf{C}),
```

then the projected scalar variable

```math
r = \mathbf{u}_k^\top \mathbf{x}
```

is Gaussian with

```math
m_k = \mathbf{u}_k^\top \mathbf{m},
\qquad
\sigma_k^2 = \mathbf{u}_k^\top \mathbf{C}\mathbf{u}_k.
```

### 6.2 Gaussian Mixture

For a Gaussian mixture, each component is projected separately and the mixture weights are preserved.

```julia
using LinearAlgebra
using Distributions
using PCDSampling

dist = MixtureModel([
    MvNormal([-1.0, 0.0], I(2)),
    MvNormal([ 1.0, 0.0], I(2)),
], [0.5, 0.5])

u = [1.0, 0.0]

proj_dist = project(dist, u)
```

If the target density is a Gaussian mixture

```math
f(\mathbf{x}) =
\sum_{i=1}^{C}
v_i \mathcal{N}(\mathbf{m}_i, \mathbf{C}_i),
```

then its projection along direction $\mathbf{u}_k$ is a one-dimensional Gaussian mixture with projected means and variances

```math
m_{k,i} = \mathbf{u}_k^\top \mathbf{m}_i,
\qquad
\sigma_{k,i}^2 = \mathbf{u}_k^\top \mathbf{C}_i \mathbf{u}_k.
```

The mixture weights $v_i$ remain unchanged.

---

## 7. Next Steps

- For installation instructions, see [Installation and Environment Setup](installation.md).

- For runnable examples, see [Running Example Code](examples.md).
  
- For reproducing paper figures and benchmark results, see [Reproducing Paper Results](reproducing_paper_results.md).
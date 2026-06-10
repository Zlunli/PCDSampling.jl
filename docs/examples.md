# Running Example Code

This document explains how to run the example scripts provided in this repository.

Before running the examples, make sure that the project environment has been set up. See Installation and Environment Setup.

---

## 1. Start Julia and Activate the Project

From the repository root, start Julia:

```bash
julia
```

Then activate the project environment:

```julia
using Pkg
Pkg.activate(".")
```

Load the package:

```julia
using PCDSampling
```

---

## 2. Load the Example Scripts

The CPU examples are defined in:

```text
scripts/examples.jl
```

The GPU examples are defined in:

```text
scripts/example_gpu.jl
```

Load the example scripts with:

```julia
include("scripts/examples.jl")
include("scripts/example_gpu.jl")
```

---

## 3. CPU Examples

### 3.1 Gaussian Example

Run the Gaussian example on the CPU:

```julia
sample_gaussian(N=500, P=100, L=-1, use_local=false)
```

Arguments:

- `N`: number of deterministic samples.
  
- `P`: number of projection directions.
  
- `L`: lookup table size. Use `L=-1` to disable the lookup table.
  
- `use_local`: if `true`, use the local update scheme; if `false`, use the global Newton-like update scheme.
  
- `T`: number of CPU threads. If omitted, the example uses `Threads.nthreads()`.
  

For example, to manually use four CPU threads:

```julia
sample_gaussian(N=500, P=100, L=-1, T=4, use_local=false)
```

### 3.2 Gaussian Mixture Example

Run the Gaussian mixture example on the CPU:

```julia
sample_mixture(C=20, N=500, P=100, L=-1, use_local=false)
```

Arguments:

- `C`: number of Gaussian mixture components.
  
- `N`: number of deterministic samples.
  
- `P`: number of projection directions.
  
- `L`: lookup table size. Use `L=-1` to disable the lookup table.
  
- `use_local`: if `true`, use the local update scheme; if `false`, use the global Newton-like update scheme.
  
- `T`: number of CPU threads. If omitted, the example uses `Threads.nthreads()`.
  

---

## 4. GPU Examples

The GPU examples require an NVIDIA GPU and a working CUDA setup through `CUDA.jl`.

### 4.1 Gaussian Example

Run the Gaussian example on the GPU:

```julia
sample_gaussian_gpu(N=500, P=100, L=128, use_local=false)
```

Arguments:

- `N`: number of deterministic samples.
  
- `P`: number of projection directions.
  
- `L`: lookup table size.
  
- `use_local`: if `true`, use the local update scheme; if `false`, use the global Newton-like update scheme.
  

Note that the current GPU implementation requires lookup tables and therefore does not support `L=-1`.

### 4.2 Gaussian Mixture Example

Run the Gaussian mixture example on the GPU:

```julia
sample_mixture_gpu(C=10, N=500, P=100, L=128, use_local=false)
```

Arguments:

- `C`: number of Gaussian mixture components.
  
- `N`: number of deterministic samples.
  
- `P`: number of projection directions.
  
- `L`: lookup table size.
  
- `use_local`: if `true`, use the local update scheme; if `false`, use the global Newton-like update scheme.
  

---

## 5. Next Steps

For lower-level API usage, see the [Usage Guide](docs/usage.md).

To reproduce the figures and benchmark results from the paper, see [Reproducing Paper Results](docs/reproducing_paper_results.md).
## 1. Overview

### 1.1 About This Repository

PCDSampling.jl

Draw deterministic samples from multivariate probability distributions.
A Python version of PCD-sampling is available at [https://github.com/KIT-ISAS/PCD_sampling_py](https://github.com/KIT-ISAS/PCD_sampling_py).

#### Examples

See the example scripts under `scripts/`.

GPU-acceleration currently only works with `CUDA.jl`.

#### Citation

```
@inproceedings{FUSION26_Prossel,
 address = {Trondheim, Norway},
 author = {Dominik Prossel and Zhilun Li and Petr Novikov and Uwe D. Hanebeck},
 booktitle = {Proceedings of the 29th International Conference on Information Fusion (FUSION 2026)},
 month = {June},
 title = {Fast Deterministic Sampling of Gaussian Mixture Densities using Projected Cumulative Distributions},
 year = {2026}
}
```

### 1.2 Example Results

### 1.3 Repository Structure

## 2. Quick Run

### 2.1 Environment Setup

#### 2.1.1 With Docker

This section describes how to run the project in a Docker-based Julia environment, for example under Windows Subsystem for Linux (WSL).

Before starting the container, make sure that the following requirements are satisfied:

1. Git is installed

   Clone this repository from GitHub:

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Docker is installed

   Make sure [Docker](https://www.docker.com/) is available from your WSL environment.

3. The GPU is visible from WSL

   Run:

   ```bash
   nvidia-smi
   ```

   This should print information about the available NVIDIA GPU.

4. The GPU is visible from Docker

   Run:

   ```bash
   docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
   ```

   If this command prints the GPU information, Docker can access the GPU correctly.

Then pull the Julia Docker image. This project can be run using the official Julia Docker image.

```bash
docker pull julia:1.12
```

To avoid reinstalling Julia packages every time the container is started, create a persistent Docker volume for the Julia depot:

```bash
docker volume create julia_depot
```

The volume will be mounted to `/depot` inside the container and used as `JULIA_DEPOT_PATH`.

From the root directory of this repository, start an interactive Julia container:

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd)":/workspace \
  -v julia_depot:/depot \
  -e JULIA_DEPOT_PATH=/depot \
  -w /workspace \
  julia:1.12
```

This command:

- mounts the current repository to `/workspace` inside the container,
- mounts the persistent Julia package cache to `/depot`,
- sets `/workspace` as the working directory,
- enables GPU access inside the container.

Inside the Julia REPL, run the following commands to install the Julia Dependencies:

```julia
using Pkg

Pkg.activate(".")
Pkg.instantiate()
Pkg.precompile()
```

Here, `Pkg.activate(".")` activates the project environment defined by `Project.toml`, `Pkg.instantiate()` installs all required dependencies, and `Pkg.precompile()` precompiles the installed packages.

#### 2.1.2 Without Docker

This section describes how to run the project directly on Windows without Docker.

Before starting the container, make sure that the following requirements are satisfied:

1. [Git](https://git-scm.com/install/windows) is installed

   Clone this repository from GitHub:

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Julia is installed

   Install Julia following the official [installation steps](https://julialang.org/downloads/). 

3. To run the GPU examples, an NVIDIA GPU and a working NVIDIA driver are required.

   Check whether Windows can see the GPU from PowerShell:

   ```bash 
   nvidia-smi
   ```

   This should print information about the available NVIDIA GPU.

Open PowerShell and clone the repository:

```powershell
git clone <repository-url>
cd <repository-name>
```

Start Julia from the repository root:

```powershell
julia
```

Inside the Julia REPL, run:

```julia
using Pkg

Pkg.activate(".")
Pkg.instantiate()
Pkg.precompile()
```

This activates the local Julia project defined by `Project.toml`, installs all required dependencies, and precompiles the packages.

### 2.2 Run Example Code

From the repository root, start Julia. Then activate the project environment:

```julia
using Pkg
Pkg.activate(".")
```

Load the package and include the example scripts:

```julia
using PCDSampling

include("scripts/examples.jl")
include("scripts/example_gpu.jl")
```

The CPU examples are defined in `scripts/examples.jl`, and the GPU examples are defined in `scripts/example_gpu.jl`.

#### 2.2.1 CPU Examples

Run the Gaussian example on the CPU:

```julia
sample_gaussian(N=500, P=100, L=-1, use_local=false) 
```

- `N` is the number of deterministic samples
- `P` is the number of projection directions
- `L` is the lookup table size, and `L=-1` disables the lookup table.
- `use_local` selects the local update scheme if set to `true`
- If `T` is omitted, the example uses `Threads.nthreads()`, i.e., the number of Julia threads available in the current Julia session. Pass `T` to manually set the number of threads.

Run the Gaussian mixture example on the CPU:

```julia
sample_mixture(C=20, N=500, P=100, L=-1, T=Threads.nthreads(), use_local=false)
```

- `C` is the number of Gaussian mixture components. 
- The remaining parameters have the same meaning as in `sample_gaussian`.

#### 2.2.2 GPU Examples

Run the Gaussian example on the GPU:

```julia
sample_gaussian_gpu(N=500, P=100, L=128, use_local=false)
```

- `N` is the number of deterministic samples
- `P` is the number of projection directions
- `L` is the lookup table size. Note that the current GPU implementation requires lookup tables and therefore does not support `L=-1`.
- `use_local` selects the local update scheme if set to `true`.

Run the Gaussian mixture example on the GPU:

```julia
sample_mixture_gpu(C=10, N=500, P=100, L=128, use_local=false)
```

- `C` is the number of Gaussian mixture components
- The remaining parameters have the same meaning as in `sample_gaussian_gpu`.

## 3. Reproducing Paper Results

## 4. Usage

### 4.1 CPU Sampling with `draw_samples`

The main CPU entry point is `draw_samples`. It generates deterministic samples from a target multivariate distribution using a given set of projection directions.

There are two ways to call `draw_samples`.

#### 4.1.1 Sampling Directly from a Target Distribution

Use this when you have a target distribution and want the package to construct all projected target distributions automatically.

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
using Distributions
using PCDSampling

dist = MvNormal([0.0, 0.0], I(2))
dirs = uniform_directions_2d(200)

X, iters = draw_samples(dist, 1000, dirs; max_iters=100, N_lut=-1)
```

The most important arguments are:

- `dist`: Target multivariate distribution.
- `N`: Number of deterministic samples.
- `dirs`: Projection directions.
- `N_lut`: Lookup table size. Use `N_lut=-1` to disable lookup tables.
- `max_iters`: Maximum number of optimization iterations.
- `eps`: Stopping threshold for the sample update.
- `use_local`: If `true`, use the local update scheme. If `false`, use the global Newton-like update scheme.
- `init_samples`: Optional initial sample positions. If omitted, initial samples are drawn randomly from `dist`.
- `verbose`: If `true`, print information about the final iteration and update norm.
- `nthreads`: Number of CPU threads used by the algorithm. If omitted, the current number of Julia threads is used.

The function returns:

- `X`: Matrix containing the deterministic samples.
- `iters`: Number of iterations performed.

#### 4.1.2 Sampling from Precomputed Projections

Use this when you have already created a `Projections` object and want to provide the initial samples yourself.

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

------

### 4.2 GPU Sampling with `draw_samples_gpu`

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
using Distributions
using PCDSampling

dist = MixtureModel([MvNormal([0.0, 0.0], I(2))])
dirs = uniform_directions_2d(200)

X, iters = draw_samples_gpu(dist, 1000, dirs; N_lut=128, max_iters=100)
```

The most important arguments are:

- `dist`: Target multivariate distribution.
- `N`: Number of deterministic samples.
- `dirs`: Projection directions.
- `N_lut`: Lookup table size. The GPU version requires lookup tables, so `N_lut` must be a positive integer.
- `max_iters`: Maximum number of optimization iterations.
- `eps`: Stopping threshold for the sample update.
- `use_local`: If `true`, use the local update scheme. If `false`, use the global Newton-like update scheme.
- `init_samples`: Optional initial sample positions. If omitted, initial samples are drawn randomly from `dist`.
- `verbose`: If `true`, print information about the final iteration and update norm.

The function returns:

- `X`: Matrix containing the deterministic samples.
- `iters`: Number of iterations performed.


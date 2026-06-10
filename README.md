[TOC]

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

3. Check if the GPU is visible from WSL

   Run:

   ```bash
   nvidia-smi
   ```

   This should print information about the available NVIDIA GPU.

4. Check if the GPU is visible from Docker

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

### 3.1 Benchmark Plots

The `benchmark/` folder contains the scripts used to reproduce the runtime benchmark results shown in the paper.

The reproduction workflow consists of two steps:

1. run the benchmark script to generate CSV result files,
2. run the plotting script to generate the corresponding figures.

The benchmark scripts compare the CPU and GPU implementations, as well as the original local update scheme and the proposed Newton-like update scheme.

The benchmark code uses a separate Julia project environment located in the `benchmark/` folder.

From the repository root, start Julia. Then activate the benchmark environment inside the Julia REPL:

```julia
using Pkg
Pkg.activate("benchmark")
Pkg.instantiate()
Pkg.precompile()
```

Before running the benchmarks, create the output folders:

```julia
mkpath("benchmark/bench_results")
mkpath("benchmark/bench_plots")
```

Then run the benchmark script:

```julia
include("benchmark/benchmarks.jl")
```

This script generates CSV result files in:

```text
benchmark/bench_results/
```

The main benchmark results are stored in:

```text
benchmarks_thresh_1e-3.csv
benchmarks_thresh_1e-3_local.csv
benchmarks_thresh_1e-3_gpu.csv
benchmarks_thresh_1e-3_local_gpu.csv
```

The fixed-iteration benchmark results are stored in:

```text
benchmarks_100_iters.csv
benchmarks_100_iters_local.csv
benchmarks_100_iters_gpu.csv
benchmarks_100_iters_local_gpu.csv
```

These CSV files are used to generate the runtime comparison plots.

After the CSV files have been generated, run the plotting script:

```julia
include("benchmark/plot_results.jl")
```

This script reads the benchmark CSV files from:

```text
benchmark/bench_results/
```

and generates plots in:

```text
benchmark/bench_plots/
```

The following plots are generated as both `.svg` and `.pdf` files:

```text
num_components.svg
num_components.pdf
dimension.svg
dimension.pdf
num_projections.svg
num_projections.pdf
num_samples.svg
num_samples.pdf
```

The plots correspond to the runtime benchmark results in the paper:

- `dimension` corresponds to Fig. 5(a).
- `num_projections` corresponds to Fig. 5(b).
- `num_samples` corresponds to Fig. 5(c).
- `num_components` is an additional benchmark plot.

The generated CSV files and plot files are also provided in the `benchmark/` folder of the `additional_paper_plots` branch of this repository:

```text
https://github.com/KIT-ISAS/PCDSampling.jl/tree/additional_paper_plots/benchmark
```

In that branch, `benchmark/bench_results/` contains the CSV result files, and `benchmark/bench_plots/` contains the generated plot files.

#### 3.1.1 Python Comparison Plot

The Julia/Python comparison plot is generated by the `create_plot_py_jl()` function in `benchmark/plot_results.jl`.

This plot compares the Julia implementation with the Python implementation and generates the following files:

```text
jl_py_num_samples.svg
jl_py_num_samples.pdf
```

To generate this plot, the Julia fixed-iteration benchmark results must first be available. These are generated by running the Julia benchmark script described above.

In addition, the following Python benchmark result file is required:

```text
benchmark/bench_results/python_results_100_iterations_gpu.csv
```

This file is not generated by the Julia benchmark script in this repository. It can be generated by running the benchmark code from the Python implementation:

```text
https://github.com/KIT-ISAS/PCD_sampling_py
```

After generating the Python benchmark result, place the CSV file at:

```text
benchmark/bench_results/python_results_100_iterations_gpu.csv
```

Then run the plotting script:

```julia
include("benchmark/plot_results.jl")
```

This will generate the Julia/Python comparison plot in:

```text
benchmark/bench_plots/
```

If the Python CSV file is not available, the Julia CPU/GPU benchmark plots can still be generated, but the Julia/Python comparison plot cannot be reproduced.

### 3.2 Additional Paper Plots

Additional plots from the paper can be reproduced from the [`additional_paper_plots` branch](https://github.com/KIT-ISAS/PCDSampling.jl/tree/additional_paper_plots).

This branch contains the plotting code, input data, and pre-generated plot files used for the additional paper figures. The relevant files are located in:

```text
paper_plots/
```

The folder contains:

```text
paper_plots/Project.toml
paper_plots/generate_plots.jl
paper_plots/data/
paper_plots/plots/
```

The `data/` folder contains input data required for generating the figures, and the `plots/` folder contains the generated `.pdf`, `.svg`, and `.png` files.

---

From the repository root, start Julia. Then activate the `paper_plots` environment inside the Julia REPL:

```julia
using Pkg
Pkg.activate("paper_plots")
Pkg.instantiate()
Pkg.precompile()
```

Then run the plotting script:

```julia
include("paper_plots/generate_plots.jl")
```

The generated figures are saved to:

```text
paper_plots/plots/
```

---

The script `generate_plots.jl` generates the following paper figures.

The function `first_page()` generates the deterministic sampling and random sampling comparison used in Fig. 1. It reads the data file

```text
paper_plots/data/2000.csv
```

to construct the target Gaussian mixture density. It then draws deterministic samples using the proposed PCD-based method and compares them with random samples from the same target density.

The generated files are:

```text
first_page.pdf
first_page.svg
first_page_random.pdf
first_page_random.svg
```

The function `slices_and_projections()` generates the slice-and-projection illustration used in Fig. 2. It constructs a two-dimensional Gaussian mixture density and visualizes the relationship between slices of the density and its one-dimensional projection.

The generated files are:

```text
slices_and_projections.pdf
slices_and_projections.png
```

The function `qualitative_comparison()` generates the qualitative comparison of different optimization methods used in Fig. 3. It compares:

- the original local update scheme,
- the proposed Newton-like update scheme,
- LCD reference samples.

The LCD samples are not computed by this repository. Instead, they are loaded from the provided data file:

```text
paper_plots/data/lcd_samples.csv
```

The generated files are:

```text
original.pdf
original.svg
proposed.pdf
proposed.svg
LCD.pdf
LCD.svg
```

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

### 4.3 Projection Directions

Both `draw_samples` and `draw_samples_gpu` require a set of projection directions. The package provides two helper functions for generating such directions.

#### 4.3.1 Uniform Directions in 2D

Use `uniform_directions_2d` to generate uniformly spaced directions in the two-dimensional plane:

```julia
dirs = uniform_directions_2d(200)
```

This creates `200` two-dimensional unit direction vectors distributed uniformly over the interval `[0, π)`. Each direction can be used to project the target distribution onto a one-dimensional axis.

Here, `200` is the number of projection directions. Increasing this number provides more projection information, but also increases the computational cost.

#### 4.3.2 Random Directions

Use `random_directions` to generate random unit direction vectors:

```julia
dirs = random_directions(dim, n_directions)
```

This creates `n_directions` random direction vectors in `dim` dimensions. Each vector is normalized to unit length, meaning that its Euclidean norm is equal to `1`.

Here:

- `dim` is the dimension of the target distribution.
- `n_directions` is the number of random projection directions.

The directions are generated by drawing Gaussian random vectors and normalizing them to unit length. This is based on the standard method by Muller (1959) for generating random directions on a sphere.

### 4.4 The `Projections` Structure

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

### 4.5 Stopping Conditions

The PCD sampling algorithm is iterative. It stops when the stopping condition returns `true`.

The package provides two helper functions for constructing stopping conditions:

- `fixed_iters`
- `max_iters_and_small_delta`

Users may also provide a custom stopping condition. A custom stopping condition should be a function that takes the current sample update as input and returns `true` when the algorithm should stop.

#### 4.5.1 Fixed Number of Iterations

Use `fixed_iters` to run the algorithm for a fixed number of iterations:

```julia
stop_cond = fixed_iters(100)

X, iters = draw_samples(dist, 1000, dirs; stop_cond=stop_cond)
```

Here, `100` is the maximum number of iterations. This stopping condition only checks the iteration count and ignores the update error.

#### 4.5.2 Maximum Iterations and Small Update

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

### 4.6 Projecting Target Distributions

The function `project` computes the one-dimensional projected target distribution along a projection direction.

The relevant function signatures are:

```julia
project(dist::AbstractMvNormal, u)
project(dist::MultivariateMixture, u)
```

It is used to construct the projected target distributions required by the PCD sampling algorithm.

#### 4.6.1 Multivariate Gaussian

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
$$
\mathbf{x} \sim \mathcal{N}(\mathbf{m}, \mathbf{C}),
$$
then the projected scalar variable
$$
r = \mathbf{u}_k^\top \mathbf{x}
$$
is Gaussian with
$$
m_k = \mathbf{u}_k^\top \mathbf{m},
\qquad
\sigma_k^2 = \mathbf{u}_k^\top \mathbf{C}\mathbf{u}_k.
$$

#### 4.6.2 Gaussian Mixture

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

# Installation and Environment Setup

This document describes how to set up the Julia environment for running `PCDSampling.jl`.

The project can be run either directly on the host system or inside a Docker container.

---

## 1. Requirements

The basic requirements are:

- Git
  
- Julia 1.11 or newer
  
- An NVIDIA GPU and working NVIDIA driver, if GPU acceleration is required
  

GPU acceleration is implemented with `CUDA.jl` and therefore requires an NVIDIA GPU.

---

## 2. Clone the Repository

Clone the repository and enter the project folder:

```bash
git clone https://github.com/KIT-ISAS/PCDSampling.jl
cd PCDSampling.jl
```

---

## 3. Native Julia Setup

Start Julia from the repository root:

```bash
julia
```

Inside the Julia REPL, activate the project environment and install the dependencies:

```julia
using Pkg

Pkg.activate(".")
Pkg.instantiate()
Pkg.precompile()
```

Here:

- `Pkg.activate(".")` activates the project environment defined by `Project.toml`.
  
- `Pkg.instantiate()` installs all dependencies listed in the project files.
  
- `Pkg.precompile()` precompiles the installed packages.
  

After this step, the package can be loaded with:

```julia
using PCDSampling
```

---

## 4. GPU Setup

To use the GPU implementation, make sure that an NVIDIA GPU is available.

Check whether the GPU is visible from the system:

```bash
nvidia-smi
```

If this command prints information about the available NVIDIA GPU, the driver is available.

Inside Julia, the GPU backend can be checked with:

```julia
using CUDA
CUDA.versioninfo()
```

The GPU examples are defined in:

```text
scripts/example_gpu.jl
```

Note that the GPU implementation currently requires lookup tables and therefore does not support `N_lut=-1`.

---

## 5. Docker Setup

A Docker-based setup is useful if you want to run the project in an isolated Julia environment, for example under Windows Subsystem for Linux (WSL).

Before starting the container, make sure Docker is installed and can access the GPU.

Check whether the GPU is visible from WSL or Linux:

```bash
nvidia-smi
```

Then check whether Docker can access the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

Pull the official Julia Docker image:

```bash
docker pull julia:1.12
```

To avoid reinstalling Julia packages every time the container is started, create a persistent Docker volume for the Julia depot:

```bash
docker volume create julia_depot
```

From the repository root, start an interactive Julia container:

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
  

Inside the Julia REPL in the container, activate the project and install the dependencies:

```julia
using Pkg

Pkg.activate(".")
Pkg.instantiate()
Pkg.precompile()
```

---

## 6. Next Steps

After the environment has been set up, continue with:

- [Running examples](examples.md)
  
- [Usage guide](usage.md)
  
- [Reproducing paper results](reproducing_paper_results.md)
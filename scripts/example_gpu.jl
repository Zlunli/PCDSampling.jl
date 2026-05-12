using PCDSampling
using Distributions
using LinearAlgebra
using CairoMakie
using CUDA

function sample_gaussian_gpu(;N=1000, L=128, use_local=false)
    dist = MixtureModel([MvNormal([0.0, 0.0], I(2))])
    @time X, iters = draw_samples_gpu(dist, N, uniform_directions_2d(200); use_local, max_iters=100, N_lut=L, verbose=true)

    f = Figure()
    ax = Axis(f[1, 1], aspect=DataAspect())
    scatter!(ax, X)
    display(f)
end

function sample_mixture_gpu(C=10; N=1000, L=128, use_local=false)
    dist = MixtureModel([MvNormal(randn(2), Diagonal(randn(2).^2)) for _ in 1:C])

    display(dist)

    @time X, iters = draw_samples_gpu(dist, N, uniform_directions_2d(200); use_local, max_iters=100, N_lut=L, verbose=true)

    f = Figure()
    ax = Axis(f[1, 1], aspect=DataAspect())
    scatter!(ax, X)
    display(f)
end
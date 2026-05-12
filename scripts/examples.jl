using PCDSampling
using Distributions
using LinearAlgebra
using CairoMakie

function sample_gaussian(;N=1000, P=200, L=-1, T=Threads.nthreads(), use_local=false)
    dist = MvNormal([0.0, 0.0], I(2))
    @time X, iters = draw_samples(dist, N, uniform_directions_2d(P); use_local, N_lut=L, verbose=false, nthreads=T)

    f = Figure()
    ax = Axis(f[1, 1], aspect=DataAspect())
    scatter!(ax, X)
    display(f)
end

function sample_mixture(;C=100, N=1000, P=200, L=-1, T=Threads.nthreads(), use_local=false)
    dist = MixtureModel([MvNormal(randn(2), Diagonal(randn(2).^2)) for _ in 1:C])
    @time X, iters = draw_samples(dist, N, uniform_directions_2d(P); use_local, N_lut=L, verbose=false, nthreads=T)

    f = Figure()
    ax = Axis(f[1, 1], aspect=DataAspect())
    scatter!(ax, X)
    display(f)
end

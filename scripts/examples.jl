using PCDSampling
using Distributions
using LinearAlgebra
using CairoMakie

using BenchmarkTools

function sample_gaussian(;N=1000, L=-1, T=Threads.nthreads(), use_local=false)
    dist = MvNormal([0.0, 0.0], I(2))
    @time X = draw_samples(dist, N, uniform_directions_2d(200); use_local, N_lut=L, verbose=false, nthreads=T)
    # @btime X = draw_samples($dist, $N, uniform_directions_2d(200); use_local=$use_local, N_lut=$L, verbose=false, nthreads=$T)

    # f = Figure()
    # ax = Axis(f[1, 1], aspect=DataAspect())
    # scatter!(ax, X)
    # display(f)
end

function sample_gaussian_mixture(;N=1000, L=-1, T=Threads.nthreads(), use_local=false)
    dist = MixtureModel([MvNormal([0.0, 0.0], I(2))])
    @time X = draw_samples(dist, N, uniform_directions_2d(200); use_local, N_lut=L, verbose=false, nthreads=T)

    f = Figure()
    ax = Axis(f[1, 1], aspect=DataAspect())
    scatter!(ax, X)
    display(f)
end

sample_gaussian()
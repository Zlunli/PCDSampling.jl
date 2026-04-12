using CairoMakie
using PCDSampling
using Distributions
using LinearAlgebra

function quick_test(;L=-1, use_local=false)
    dist = MixtureModel([MvNormal([0.0, 0.0], I(2))])
    @time X = draw_samples(dist, 1000, uniform_directions_2d(100); use_local, N_lut=L, verbose=false)

    f = Figure()
    ax = Axis(f[1, 1], aspect=DataAspect())
    scatter!(ax, X)
    display(f)
end
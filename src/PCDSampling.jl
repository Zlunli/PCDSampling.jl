module PCDSampling

using Distributions
using OhMyThreads

include("lookup_table.jl")
include("projections.jl")
include("sampling.jl")

function fixed_iters(max_iters)
    iterator = Iterators.Stateful(1:max_iters)
    _ -> popfirst!(iterator) >= max_iters
end

function max_iters_and_small_delta(max_iters, eps)
    iterator = Iterators.Stateful(1:max_iters)
    # delta -> popfirst!(iterator) >= max_iters || norm(delta) < eps
    delta -> popfirst!(iterator) >= max_iters || maximum(abs.(delta)) < eps
end

function draw_samples(dist::MultivariateDistribution, N, dirs; max_iters=100, eps=1e-6, stop_cond=nothing, init_samples=nothing, verbose=false)
    targets = project.(Ref(dist), dirs)
    if !isa(dirs, Matrix)
        dirs = collect(dirs)
    end
    projections = Projections(targets, dirs)

    draw_samples(projections, N; max_iters, eps, stop_cond, init_samples, verbose)
end

function draw_samples(projections::Projections, N; max_iters=100, eps=1e-6, stop_cond=nothing, init_samples=nothing, verbose=false)
    if isnothing(init_samples)
        init_samples = rand(dist, N)
    end
    if isnothing(stop_cond)
        stop_cond = max_iters_and_small_delta(max_iters, eps)
    end
    pcd_sample(projections, init_samples, stop_cond; verbose)
end

end # module PCDSampling

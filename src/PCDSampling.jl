module PCDSampling

using LinearAlgebra
using Distributions
using OhMyThreads

include("lookup_table.jl")
export create_lut
include("projections.jl")
export Projections, project, uniform_directions_2d, random_directions, get_projs, get_dirs
include("sampling.jl")
export cvm_grad_hess

export fixed_iters, max_iters_and_small_delta, draw_samples

function fixed_iters(max_iters)
    iterator = Iterators.Stateful(0:max_iters)
    _ -> popfirst!(iterator) >= max_iters
end

function max_iters_and_small_delta(max_iters, eps)
    iterator = Iterators.Stateful(0:max_iters)
    # delta -> popfirst!(iterator) >= max_iters || norm(delta) < eps
    delta -> popfirst!(iterator) >= max_iters || maximum(abs.(delta)) < eps
end

function draw_samples(dist::MultivariateDistribution, N, dirs;
                        use_local=false, N_lut=-1, max_iters=100, eps=1e-6, stop_cond=nothing,
                        init_samples=nothing, verbose=false)
    if N_lut == -1
        targets = project.(Ref(dist), dirs)
    else
        targets = create_lut.(project.(Ref(dist), dirs), N_lut)
    end
    if !isa(dirs, Matrix)
        dirs = reduce(hcat, dirs)
    end
    projections = Projections(targets, dirs)

    if isnothing(init_samples)
        init_samples = rand(dist, N)
    end
    draw_samples(projections, init_samples; use_local, max_iters, eps, stop_cond, verbose)
end

function draw_samples(projections::Projections, init_samples; max_iters=100, eps=1e-6, stop_cond=nothing, use_local=false, verbose=false)
    if isnothing(stop_cond)
        stop_cond = max_iters_and_small_delta(max_iters, eps)
    end
    pcd_sample(projections, init_samples, stop_cond; use_local, verbose)[1]
end

end # module PCDSampling

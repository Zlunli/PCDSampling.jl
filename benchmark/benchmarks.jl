using PCDSampling
using BenchmarkTools
using Random, Distributions, LinearAlgebra, Statistics
using CSV
using Tables
using CUDA

function build_runs_from_grids(grids)
    runs = Vector{NTuple{4, Int64}}()
    for grid in grids
        n_samples, n_components, n_dims, n_projs = grid
        for N in n_samples
            for C in n_components
                for D in n_dims
                    for P in n_projs
                        push!(runs, (N, C, D, P))
                    end
                end
            end
        end
    end
    return runs
end

function benchmark()
    # grid = [[[20, 50, 100, 200, 500, 1000, 2000, 3000, 5000], [1], [2], [1000]]]
    
    # grid = [[[20, 50, 100, 200, 500, 1000, 3000, 5000], [200], [2], [1000]],
    #             [[2000], [200], [4, 6, 8, 10], [1000]],
    #             [[2000], [200], [2], [100, 200, 500, 2000]],
    #             [[2000], [200], [2], [1000]]]

    grids =[[[20, 50, 100, 200, 1000], [200], [2], [1000]],
                [[500], [200], [4, 6, 8, 10], [1000]],
                [[500], [200], [2], [100, 200, 500, 2000]],
                [[500], [200], [2], [1000]]]
                
    runs = build_runs_from_grids(grids)
    do_benchmark(runs, use_gpu=false, use_local=false)
    do_benchmark(runs, use_gpu=false, use_local=true)
    do_benchmark(runs, use_gpu=true, use_local=false)
    do_benchmark(runs, use_gpu=true, use_local=true)
end

function test_benchmark()
    runs = build_runs_from_grids([[[100], [100], [4], [200]]])
    do_benchmark(runs, use_gpu=false, use_local=false)
    do_benchmark(runs, use_gpu=false, use_local=true)
    do_benchmark(runs, use_gpu=true, use_local=false)
    do_benchmark(runs, use_gpu=true, use_local=true)
end

function do_benchmark(runs; max_iters=5000, eps=1e-3, use_local=false, use_gpu=false, n_repeats=40, result_path="./benchmark/bench_results", filename="benchmarks_thresh_1e-3")
    Random.seed!(42)
    if use_local
        filename *= "_local"
    end
    if use_gpu
        filename *= "_gpu"
    end

    filepath = joinpath(result_path, filename*".csv")
    CSV.write(filepath, [], header=["n_samples", "n_components", "dims", "n_projs", ("iters_$i" for i in 1:n_repeats)..., ("run_$i" for i in 1:n_repeats)...], delim=",")

    for run in runs
        N, C, D, P = run
        println("N: $N, C: $C, D: $D, P: $P")
        dirs = random_directions(D, P)

        all_times = -1*ones(n_repeats)
        all_iters = -1*ones(n_repeats)
        total_elapsed = 0
        
        # warmup
        target = MixtureModel([MvNormal(randn(D), Diagonal(randn(D).^2)) for _ in 1:C])
        t = @elapsed _, iters = draw_samples(target, N, dirs; max_iters, eps, use_local, N_lut=128, verbose=false)
        
        for i in 1:n_repeats
            target = MixtureModel([MvNormal(randn(D), Diagonal(randn(D).^2)) for _ in 1:C])
            if use_gpu
                t = @elapsed _, iters = draw_samples_gpu(target, N, dirs; max_iters, eps, use_local, N_lut=128, verbose=false)
            else
                t = @elapsed _, iters = draw_samples(target, N, dirs; max_iters, eps, use_local, N_lut=128, verbose=false)
            end
            all_times[i] = t
            all_iters[i] = iters
            total_elapsed += t
            if total_elapsed > 180
                break
            end
        end
        
        CSV.write(filepath, Tables.table(reshape([[N, C, D, P]; all_iters; all_times], 1, :)), delim=",", append=true) #times are in nanoseconds
    end
end

function test_gauss_pcd()
    # LinearAlgebra.BLAS.set_num_threads(1) # otherwise we get weird performance results when using multiple threads for the gradient computation
    # display(LinearAlgebra.BLAS.get_config())
    MKL.set_num_threads(1)
end

# benchmark()
test_benchmark()
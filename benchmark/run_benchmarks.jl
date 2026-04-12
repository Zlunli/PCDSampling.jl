using PCDSampling
using BenchmarkTools
using Random, Distributions, LinearAlgebra, Statistics
using CSV
using Tables

function benchmark()
    # n_samples = [20, 50, 100, 200, 500, 1000, 2000, 3000, 5000]
    # n_components = [2, 5, 10, 20, 50, 100, 200]
    # n_dims = [2, 4, 6, 8, 10]
    # # n_dims = [2, 10]
    # n_projs = [100, 200, 500, 1000, 2000]

    # n_samples = [20, 50, 100, 200, 500, 1000, 2000, 3000, 5000]
    # n_components = [2, 5, 10, 20, 50, 100, 200]
    # n_dims = [2, 4, 6, 8, 10]
    # # n_dims = [2, 10]
    # n_projs = [100, 200, 500, 1000, 2000]
    # do_benchmark([[n_samples, n_components, n_dims, n_projs]])
    
    # grid = [[[20, 50, 100, 200, 500, 1000, 2000, 3000, 5000], [1], [2], [1000]]]
    
    # grid = [[[20, 50, 100, 200, 500, 1000, 3000, 5000], [200], [2], [1000]],
    #             [[2000], [200], [4, 6, 8, 10], [1000]],
    #             [[2000], [200], [2], [100, 200, 500, 2000]],
    #             [[2000], [200], [2], [1000]]]

    grid =[[[20, 50, 100, 200, 1000], [200], [2], [1000]],
                [[500], [200], [4, 6, 8, 10], [1000]],
                [[500], [200], [2], [100, 200, 500, 2000]],
                [[500], [200], [2], [1000]]]

    do_benchmark_local(grid)
    do_benchmark_lut(grid)
end

function test_benchmark()
    # do_benchmark([5000], [2], [2], [1500])
    # do_benchmark([[[3000], [200], [10], [2000]]])
    do_benchmark_lut([[[100], [100], [4], [200]]])
    do_benchmark_local([[[100], [100], [4], [200]]])
end

function do_benchmark_lut(grids; steps=100, n_runs=40, result_path="./bench_results")
    Random.seed!(42)
    filepath = joinpath(result_path, "benchmarks_thresh_1e-3_new.csv")
    CSV.write(filepath, [], header=["n_samples", "n_components", "dims", "n_projs", ("iters_$i" for i in 1:n_runs)..., ("run_$i" for i in 1:n_runs)...], delim=",")

    for grid in grids
        n_samples, n_components, n_dims, n_projs = grid
        for N in n_samples
            for C in n_components
                for D in n_dims
                    for P in n_projs
                        println("N: $N, C: $C, D: $D, P: $P")
                        dirs = random_directions(D, P)

                        all_times = -1*ones(n_runs)
                        all_iters = -1*ones(n_runs)
                        total_elapsed = 0
                        
                        # warmup
                        target = MixtureModel([MvNormal(randn(D), Diagonal(randn(D).^2)) for _ in 1:C])
                        init=rand(target, N)
                        t = @elapsed _, iters = PCD.pcd_gm_parallel_dirs_x_lut_wrap(target, init, dirs, 
                                max_iters_and_small_delta(5001, 1e-3), verbose=false)
                                # fixed_iters(steps+1), verbose=false)
                        for i in 1:n_runs
                            target = MixtureModel([MvNormal(randn(D), Diagonal(randn(D).^2)) for _ in 1:C])
                            init=rand(target, N)
                            t = @elapsed _, iters = PCD.pcd_gm_parallel_dirs_x_lut_wrap(target, init, dirs, 
                                max_iters_and_small_delta(5001, 1e-3), verbose=false)
                                # fixed_iters(steps+1), verbose=false)

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
            end
        end
    end
end

function do_benchmark_local(grids; steps=100, n_runs=40, result_path="./bench_results")
    Random.seed!(42)
    filepath = joinpath(result_path, "benchmarks_thresh_1e-3_local_new.csv")
    CSV.write(filepath, [], header=["n_samples", "n_components", "dims", "n_projs", ("iters_$i" for i in 1:n_runs)..., ("run_$i" for i in 1:n_runs)...], delim=",")

    for grid in grids
        n_samples, n_components, n_dims, n_projs = grid
        for N in n_samples
            for C in n_components
                for D in n_dims
                    for P in n_projs
                        println("N: $N, C: $C, D: $D, P: $P")
                        dirs = random_directions(D, P)

                        all_times = -1*ones(n_runs)
                        all_iters = -1*ones(n_runs)
                        total_elapsed = 0
                        
                        # warmup
                        target = MixtureModel([MvNormal(randn(D), Diagonal(randn(D).^2)) for _ in 1:C])
                        init=rand(target, N)
                        t = @elapsed _, iters = PCD.draw_samples(target, init, dirs, 
                                max_iters_and_small_delta(5001, 1e-3), verbose=false, use_local=true)
                                # fixed_iters(steps+1), verbose=false)
                        for i in 1:n_runs
                            target = MixtureModel([MvNormal(randn(D), Diagonal(randn(D).^2)) for _ in 1:C])
                            init=rand(target, N)
                            t = @elapsed _, iters = draw_samples(target, init, dirs, 
                                max_iters_and_small_delta(5001, 1e-3), verbose=false, use_local=true)
                                # fixed_iters(steps+1), verbose=false)

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
            end
        end
    end
end

function test_result()
    C=200
    # target = MixtureModel([MvNormal([-1.0, 0.0], Diagonal([3.0, 1.0])), MvNormal([1.0, 0.0], Diagonal([1.0, 1.0]))])
    target = MixtureModel([MvNormal(m, Diagonal(s .* [1.0, 1.0])) for (m,s) in zip(eachcol(randn(2, C)), rand(C))])
    # PCD.test_gauss_pcd(D=2, N=5000, dirs=equal_partitions_2d(2000), target=target, plot=true)
    test_gauss_pcd(D=2, N=500, dirs=equal_partitions_2d(100), target=target, plot=true)
    # PCD.test_gauss_pcd(D=2, N=100, dirs=equal_partitions_2d(400), plot=true, target=target)
end

function test_gauss_pcd(;D=2, N=200, dirs=equal_partitions_2d(200), target=nothing)
    Random.seed!(17)

    # LinearAlgebra.BLAS.set_num_threads(1) # otherwise we get weird performance results when using multiple threads for the gradient computation
    # display(LinearAlgebra.BLAS.get_config())
    MKL.set_num_threads(1)

    if isnothing(target)
        #target = MvNormal(Diagonal(fill(1.0, D))) 
        #target = MvNormal([0.0, 0.0], Diagonal([3.0, 1.0]))
        # target = MixtureModel([MvNormal([-1.0, 0.0], Diagonal([3.0, 1.0])), MvNormal([1.0, 0.0], Diagonal([1.0, 1.0]))])
        # target = MixtureModel([MvNormal([-1.0, 0.0], Diagonal([1.0, 1.0])), MvNormal([1.0, 0.0], Diagonal([1.0, 1.0]))])
        target = MixtureModel([MvNormal(fill(0.0, D), Diagonal(fill(1.0, D))), MvNormal(fill(0.0, D), Diagonal(fill(1.0, D)))])
        # target = MixtureModel([MvNormal(randn(2), Diagonal([1.0, 1.0])) for _ in 1:50])
        #init = vcat([[i j] for i in -2:1, j in -2:1]...)' / 10
    end
    init = randn(Float64, D, N)

    #result = pcd(target, copy(init), dirs, max_iters_and_small_delta(8000, 1e-8))
    # @time result = pcd(target, copy(init), dirs, max_iters_and_small_delta(8000, N*1e-8))
    # @time result = pcd_gm(target, copy(init), dirs, max_iters_and_small_delta(300, N*1e-8))
    # @time result = pcd_gm_threaded_dirs(target, copy(init), dirs, max_iters_and_small_delta(300, N*1e-8))
    # @time result = pcd_gm_parallel_dirs_x_wrap(target, copy(init), dirs, max_iters_and_small_delta(300, N*1e-8))
    # @time result = pcd_gm_parallel_dirs_x_lut_wrap(target, copy(init), dirs, max_iters_and_small_delta(300, N*1e-8))[1] # should be fastest
    @time result = pcd_gm_parallel_dirs_x_lut_wrap(target, copy(init), dirs, fixed_iters(300))[1] # should be fastest
    # @time result = pcd_gm_parallel_dirs_x_local_wrap(target, copy(init), dirs, max_iters_and_small_delta(100, N*1e-8))
    #@time result = pcd_ls(target, copy(init), dirs, max_iters_and_small_delta(1000, 1e-8))
    #@time result = pcd_gpu(target, copy(init), dirs, max_iters_and_small_delta(10000, 1e-8))
    result
end

benchmark()
# test_benchmark()
# test_result()
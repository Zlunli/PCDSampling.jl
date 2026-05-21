module PCDSampling

# 这里主要可能用到 norm、矩阵/向量运算、线性代数相关函数
using LinearAlgebra
# 用于概率分布类型，例如 MultivariateDistribution
using Distributions

# 引入 OhMyThreads.jl
# 这里主要用到 tcollect，用于多线程并行 collect
# 例如对多个 projection direction 并行计算目标投影分布
using OhMyThreads

include("lookup_table.jl")
export create_lut
include("projections.jl")
export Projections, project, uniform_directions_2d, random_directions, get_projs, get_dirs
include("sampling.jl")
export cvm_grad_hess

export fixed_iters, max_iters_and_small_delta, draw_samples, draw_samples_gpu

"""
创建一个停止条件函数
 - 这个停止条件只看迭代次数，不看当前更新量 delta。
 - 返回 True 表示迭代应该停止
 """
function fixed_iters(max_iters)
    iterator = Iterators.Stateful(0:max_iters)
    _ -> popfirst!(iterator) >= max_iters
end

"""
创建一个停止条件函数。它有两个停止标准：
  1. 迭代次数达到 max_iters
  2. 当前更新量 delta 已经足够小
"""
function max_iters_and_small_delta(max_iters, eps)
    iterator = Iterators.Stateful(0:max_iters)

    # delta -> popfirst!(iterator) >= max_iters || norm(delta) < eps # Check the Euclidean norm, norm(delta)
    delta -> popfirst!(iterator) >= max_iters || maximum(abs.(delta)) < eps # Check the infinity norm
end

"""
主接口函数。
从多维分布 dist 中生成 N 个 deterministic samples，使这些 samples 在多个 projection directions 上尽量匹配目标分布。
  - dist::MultivariateDistribution 多维分布类型
  - N: 表示希望生成多少个样本点
  - dirs: 表示投影方向。它可以是矩阵，也可以是向量列表。
  - use_local: 该参数会传递给底层的 pcd_sample
  - N_lut: 这个参数控制是否使用 lookup table
      - -1 表示不建立lookup table
      - >0 表示对每个一维投影目标分布建立一个 lookup table
  - max_iters=100 默认最大迭代次数是 100
  - eps=1e-6: 默认收敛阈值
  - stop_cond=nothing 用户可以自行传入停止条件，但是需要满足，如果停止返回true
  - init_samples=nothing 初始样本点，如果用户没有提供初始样本，程序从原始分布中随机抽取 N 个样本作为初始值。
  - verbose=false 是否输出更多迭代过程信息
  - nthreads=Threads.nthreads() 使用多少个线程。默认使用 Julia 当前可用的线程数。
"""
function draw_samples(dist::MultivariateDistribution, N, dirs;
                        use_local=false, N_lut=-1, max_iters=100, eps=1e-6, stop_cond=nothing,
                        init_samples=nothing, verbose=false, nthreads=Threads.nthreads())
    # 整理投影方向
    ## 如果 dirs 不是矩阵，就把它转换成矩阵，并且让每一列代表一个方向。
    if !isa(dirs, Matrix)
        dirs = reduce(hcat, dirs)
    end
    
    # 构造目标投影分布 targets
    # 对每一个投影方向 d，把原始多维分布 dist 投影成一个一维分布。
    ## 不使用 lookup table 的情况
    if N_lut == -1 
        targets = tcollect(project(dist, d) for d in eachcol(dirs))
    ## 使用 lookup table 的情况
    else 
        # 每个目标投影分布不会直接保存原始分布对象，而是转换成 lookup table。
        targets = tcollect(LookupTable, create_lut(project(dist, d), N_lut) for d in eachcol(dirs))
    end

    # 构造 Projections 对象
    ## 每个方向 d_k 以及该方向上的目标一维投影分布 target_k
    projections = Projections(targets, dirs)

    # 初始化样本
    # 如果用户没有提供初始样本，那么程序用原始分布 dist 随机生成 N 个样本作为初始点。
    if isnothing(init_samples)
        init_samples = rand(dist, N)
    end
    draw_samples(projections, init_samples; use_local, max_iters, eps, stop_cond, verbose, nthreads)
end

function draw_samples(projections::Projections, init_samples; max_iters=100, eps=1e-6, stop_cond=nothing,
                        use_local=false, verbose=false, nthreads=Threads.nthreads())

    # 如果用户没有提供停止条件，就创建默认停止条件
    if isnothing(stop_cond)
        stop_cond = max_iters_and_small_delta(max_iters, eps)
    end

    # 调用核心函数
    pcd_sample(projections, init_samples, stop_cond; use_local, verbose, nthreads)
end

function draw_samples_gpu(dist::MultivariateDistribution, N, dirs; use_local=false, N_lut=-1, 
                max_iters=100, eps=1e-6, stop_cond=nothing, init_samples=nothing, verbose=false)
    ext = Base.get_extension(@__MODULE__, :PCDSamplingCUDAExt)
    ext === nothing && error("CUDA extension not loaded. Try 'using CUDA'.")
    return ext.draw_samples_gpu(dist, N, dirs; use_local, N_lut, max_iters, eps, stop_cond, init_samples, verbose)
end

end # module PCDSampling

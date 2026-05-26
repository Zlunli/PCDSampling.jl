# 计算某个一维投影位置上的 CvM 局部误差项和 PDF 项
"""
对普通一维分布计算 CvM 更新所需的两个局部量

dist::UnivariateDistribution:
  某个投影方向上的目标一维分布

x:
  当前某个 Dirac sample 在该方向上的投影值

i:
  当前 sample 在该方向投影排序后的 rank
  也就是它是第几个投影点

N:
  sample 总数

返回：
  1. dirac_cdf - cdf(dist, x)
     即 empirical Dirac CDF 和目标 CDF 的差值

  2. pdf(dist, x)
     即目标分布在 x 处的 PDF 值

这两个量会被后面的 Newton/local update 用来构造更新方向和 Hessian 近似
"""
function cvm_grad_hess(dist::UnivariateDistribution, x, i, N)
    # 对第 i 个排序后的 sample，经验 CDF 使用中点近似：
    #
    # F_D(x_i) ≈ (i - 0.5) / N
    #
    # 这里不用 i/N，而用 (i-0.5)/N，是为了表示阶梯 CDF 在跳跃处的中间位置
    dirac_cdf = (i-0.5)/N

    # 返回两个量：
    #
    # c = F_D(x) - F_target(x)
    # p = f_target(x)
    #
    # 其中：
    # cdf(dist, x) 是目标投影分布的 CDF
    # pdf(dist, x) 是目标投影分布的 PDF
    dirac_cdf - cdf(dist, x), pdf(dist, x)
end

# 使用类似 Newton / Gauss-Newton 的方式更新每个 sample
"""
对所有 samples 做一次 Newton-style 更新
函数名最后的 ! 是 Julia 习惯：
表示这个函数会原地修改输入变量

这里会修改：
  X
  delta_x

输入参数：

X:
  当前 sample 位置矩阵
  size(X) = dim × N
  每一列 X[:, i] 是第 i 个 sample

delta_x:
  保存每个 sample 的更新量
  size(delta_x) = dim × N
  每一列 delta_x[:, i] 是第 i 个 sample 的更新步长

projections:
  Projections 对象
  包含所有目标投影分布和对应方向

proj_X:
  当前所有 samples 在所有方向上的投影值
  size(proj_X) = N × M
  proj_X[i, m] 表示第 i 个 sample 在第 m 个方向上的投影值

proj_rank:
  当前所有 samples 在所有方向上的排序 rank
  size(proj_rank) = N × M
  proj_rank[i, m] 表示第 i 个 sample 在第 m 个方向上的排序名次

nthreads:
  使用多少线程并行
"""
function netwon_step!(X, delta_x, projections, proj_X, proj_rank; nthreads=Threads.nthreads())

    @tasks for i in axes(X, 2)

        # 设置并行任务数量
        @set ntasks=nthreads     
        
        @local begin
            local_delta = zeros(eltype(X), size(X, 1))
            hess_x = zeros(eltype(X), size(X, 1), size(X, 1))
        end

        # 每次处理一个新的 sample i 时，把局部梯度和 Hessian 清零
        local_delta .= 0.0
        hess_x .= 0.0
        

        # 遍历所有投影方向
        #
        # enumerate(projections) 会依次返回：
        #
        # m:
        #   当前是第几个投影方向
        #
        # (target, dir):
        #   target 是该方向上的目标一维分布或 LookupTable
        #   dir 是该投影方向向量
        @inbounds for (m, (target, dir)) in enumerate(projections)

            # 计算当前 sample i 在方向 m 上的 CvM 局部误差量
            #
            # proj_X[i, m]:
            #   第 i 个 sample 在第 m 个方向上的投影值
            #
            # proj_rank[i, m]:
            #   第 i 个 sample 在第 m 个方向上的排序 rank
            #
            # size(X, 2):
            #   sample 总数 N
            #
            # 返回：
            #   step:
            #       F_D(x) - F_target(x)
            #
            #   hess_step:
            #       f_target(x)
            step, hess_step = cvm_grad_hess(target, proj_X[i, m], proj_rank[i, m], size(X, 2))
            
            # 把一维方向上的误差 step 反投影回原始 dim 维空间
            #
            # dir[j] 是第 m 个投影方向在第 j 个坐标上的分量
            #
            # local_delta[j] += dir[j] * step
            #
            # 数学上相当于累加：
            #
            # local_delta += step * dir
            #
            # 如果 step 表示一维投影上的 CDF 差值，
            # 那么乘以 dir 就把这个一维误差信息映射回高维 sample 坐标
            for j in eachindex(local_delta)
                local_delta[j] += dir[j] * step
            end
            
            # 累积 Hessian 近似
            #
            # hess_step 是目标投影 PDF：
            #
            # hess_step = f_target(proj_X[i, m])
            #
            # 方向贡献是外积：
            #
            # dir * dir'
            #
            # 因此每个方向贡献：
            #
            # hess_step * dir * dir'
            #
            # 这里只填充 Hessian 的下三角部分
            # 因为 Hessian 是对称矩阵
            for j in axes(hess_x, 2)
                for k in j:size(hess_x, 1)
                    hess_x[k, j] += hess_step * dir[j] * dir[k]
                end
            end
        end
        
        # 把 hess_x 的下三角部分解释成一个对称矩阵
        #
        # :L 表示使用 lower triangle
        #
        # 也就是说，虽然前面只填了 hess_x[k, j] where k >= j，
        # 这里会把它看作完整对称矩阵
        hess_x_sym = Symmetric(hess_x, :L)

        # 对 Hessian 做 Cholesky 分解
        #
        # 如果 Hessian 是正定矩阵 H，
        # Cholesky 分解得到：
        #
        # H = L L'
        #
        # cholesky! 表示原地分解，可能会覆盖 hess_x_sym 内部数据
        fac = cholesky!(hess_x_sym)

        # 解线性方程：
        #
        # H * delta = local_delta
        #
        # 把解写入 delta_x[:, i]
        #
        # 这里 ldiv! 是原地线性求解
        #
        # @view(delta_x[:, i]) 表示第 i 个 sample 的更新向量
        ldiv!(@view(delta_x[:, i]), fac, local_delta)

        @views X[:, i] .+= delta_x[:, i]
    end
end

function local_update!(X, delta_x, projections, proj_X, proj_rank; nthreads=Threads.nthreads())
    # 对每个 sample 并行更新
    @tasks for i in axes(X, 2)
        @set ntasks=nthreads 
        
        delta_x[:, i] .= 0.0

        # 遍历所有投影方向
        @inbounds for (m, (target, dir)) in enumerate(projections)

            # 计算当前 sample 在当前方向上的一维误差项和 PDF 项
            step, hess_step = cvm_grad_hess(target, proj_X[i, m], proj_rank[i, m], size(X, 2))
            
            # max(hess_step, 1e-3) 是为了避免除以太小的 PDF
            for j in axes(delta_x, 1)
                delta_x[j, i] += dir[j] * (step / max(hess_step, 1e-3))
            end
        end
        # 对所有方向的更新取平均
        delta_x[:, i] ./= length(projections)
        @views X[:, i] .+= delta_x[:, i]
    end
end

function pcd_sample(projections::Projections, init_samples, stop_condition; use_local=false, verbose=true, nthreads=Threads.nthreads())
    
    # X 是当前样本点矩阵
    X = init_samples

    # 取出所有投影方向
    directions = get_dirs(projections)

    # 初始化 delta_x
    # 它保存每个 sample 的更新量
    delta_x = ones(eltype(X), size(X))

    # proj_X 保存所有 sample 在所有方向上的投影值
    proj_X = zeros(eltype(X), size(X, 2), length(projections))

    # proj_sp 保存每个方向上的排序 permutation
    # sp 可以理解为 sorted permutation
    proj_sp = zeros(Int, size(X, 2), length(projections))

    proj_rank = zeros(Int, size(X, 2), length(projections))

    # TODO: For weighted samples maintain cumsum of weights instead of sample rank
    # 当前代码假设所有 Dirac samples 权重相同，都是 1/N
    # 所以 empirical CDF 可以直接用 rank:
    # 如果以后支持 weighted samples，则不能只用 rank，
    # 而应该按照排序后的权重做 cumulative sum。

    # 初始化所有方向上的投影值、排序 permutation 和 rank
    for (i, dir) in enumerate(directions)
        mul!(@view(proj_X[:, i:i])', dir', X)
        sortperm!(@view(proj_sp[:, i]), @view(proj_X[:, i]))

        proj_rank[:, i] .= invperm(@view(proj_sp[:, i]))
    end

    # 记录迭代次数
    iters = 0

    while !stop_condition(delta_x)

        @tasks for i in eachindex(directions)
            @set ntasks=nthreads           

            mul!(@view(proj_X[:, i])', directions[i]', X)

            sp = @view(proj_sp[:, i])
            for j in eachindex(sp)
                k = 0
                while j+k < length(sp) && j+k > 0 && proj_X[sp[j+k], i] > proj_X[sp[j+k+1], i]
                    sp[j+k], sp[j+k+1] = sp[j+k+1], sp[j+k]
                    proj_rank[sp[j+k], i] -= 1
                    proj_rank[sp[j+k+1], i] += 1
                    k -= 1
                end
            end
        end

        # 根据参数选择 sample 更新方式
        if use_local
            local_update!(X, delta_x, projections, proj_X, proj_rank; nthreads)
        else
            netwon_step!(X, delta_x, projections, proj_X, proj_rank; nthreads)
        end
        iters += 1
    end

    if verbose
        println("final iteration: $iters")
        println("final delta norm: $(norm(delta_x))")
    end
    return X, iters
end

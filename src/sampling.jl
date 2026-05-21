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
    # 使用 OhMyThreads 的 @tasks 并行遍历每一个 sample
    #
    # axes(X, 2) 表示 X 的第 2 维索引
    # 因为 X 是 dim × N，所以 axes(X, 2) 就是 1:N
    #
    # 换句话说：
    #   for i in axes(X, 2)
    #
    # 表示对每一个 sample i 做更新
    @tasks for i in axes(X, 2)

        # 设置并行任务数量
        @set ntasks=nthreads     
        
        # 为每个 task 创建局部变量
        #
        # local_delta:
        #   当前 sample 的梯度方向累积量，大小是 dim
        #
        # hess_x:
        #   当前 sample 的 Hessian 近似矩阵，大小是 dim × dim
        #
        # @local 的作用是：
        #   在线程/任务内部复用这些局部数组，避免每次循环都重新分配内存
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

# 使用更局部、更简单的近似方式更新每个 sample
"""
对所有 samples 做一次 local update

这个版本比 netwon_step! 更简单：

它不构造完整 dim × dim Hessian，
而是在每个投影方向上用 step / hess_step 做一维缩放，
再沿着投影方向反投影回高维。

输入参数含义与 netwon_step! 基本相同。
"""
function local_update!(X, delta_x, projections, proj_X, proj_rank; nthreads=Threads.nthreads())
    # 对每个 sample 并行更新
    @tasks for i in axes(X, 2)
        @set ntasks=nthreads 
        
        # 清空第 i 个 sample 的更新量
        delta_x[:, i] .= 0.0

        # 遍历所有投影方向
        @inbounds for (m, (target, dir)) in enumerate(projections)

            # 计算当前 sample 在当前方向上的一维误差项和 PDF 项
            step, hess_step = cvm_grad_hess(target, proj_X[i, m], proj_rank[i, m], size(X, 2))
            
            # 沿着当前投影方向 dir 累加更新量
            #
            # step / hess_step 类似一维 Newton step
            #
            # max(hess_step, 1e-3) 是为了避免除以太小的 PDF
            # 如果 hess_step 很小，直接除会导致更新量爆炸
            for j in axes(delta_x, 1)
                delta_x[j, i] += dir[j] * (step / max(hess_step, 1e-3))
            end
        end
        # 对所有方向的更新取平均
        #
        # 如果有 M 个投影方向，那么：
        #
        # delta_x[:, i] = delta_x[:, i] / M
        delta_x[:, i] ./= length(projections)
        @views X[:, i] .+= delta_x[:, i]
    end
end

# 主迭代循环：投影 → 排序 → 计算 rank → 更新样本 → 判断停止
"""
PCD 主迭代函数

projections:
  Projections 对象
  包含所有目标投影分布和对应方向

init_samples:
  初始样本矩阵
  size(init_samples) = dim × N

stop_condition:
  停止条件函数
  每次迭代前会调用 stop_condition(delta_x)

use_local:
  是否使用 local_update!
  如果 false，则使用 netwon_step!

verbose:
  是否打印最终迭代次数和最终 delta norm

nthreads:
  使用多少线程

返回：
  X:
      最终 sample 矩阵
  iters:
      实际迭代次数
"""
function pcd_sample(projections::Projections, init_samples, stop_condition; use_local=false, verbose=true, nthreads=Threads.nthreads())
    
    # X 是当前样本点矩阵
    #
    # 注意：这里不是复制，而是直接引用 init_samples
    #
    # 所以后面对 X 的原地修改，也会修改传入的 init_samples
    X = init_samples

    # 取出所有投影方向
    #
    # get_dirs(projections) 在 projections.jl 中定义为：
    #
    # get_dirs(projs::Projections) = eachcol(projs.dirs)
    #
    # 所以 directions 是一个按列访问方向的迭代对象
    directions = get_dirs(projections)

    # 初始化 delta_x
    #
    # 它保存每个 sample 的更新量
    #
    # size(delta_x) = size(X) = dim × N
    #
    # 初始设为全 1，是为了保证第一次 stop_condition(delta_x) 不会因为 delta 太小而停止
    delta_x = ones(eltype(X), size(X))

    # proj_X 保存所有 sample 在所有方向上的投影值
    #
    # size(proj_X) = N × M
    #
    # 其中：
    #   N = size(X, 2) 是 sample 数量
    #   M = length(projections) 是投影方向数量
    #
    # proj_X[i, m] 表示第 i 个 sample 在第 m 个方向上的投影值
    proj_X = zeros(eltype(X), size(X, 2), length(projections))

    # proj_sp 保存每个方向上的排序 permutation
    #
    # sp 可以理解为 sorted permutation
    #
    # proj_sp[:, m] 是第 m 个方向上的排序索引
    #
    # 如果：
    #   proj_sp[:, m] = [3, 1, 2]
    #
    # 表示在第 m 个方向上：
    #   第 3 个 sample 的投影最小
    #   第 1 个 sample 的投影第二小
    #   第 2 个 sample 的投影最大
    proj_sp = zeros(Int, size(X, 2), length(projections))

    # proj_rank 保存每个 sample 在每个方向上的 rank
    #
    # proj_rank[i, m] 表示第 i 个 sample 在第 m 个方向上的排序名次
    #
    # 它是 proj_sp[:, m] 的 inverse permutation
    #
    # 例如：
    #   proj_sp[:, m] = [3, 1, 2]
    #
    # 则：
    #   sample 3 的 rank 是 1
    #   sample 1 的 rank 是 2
    #   sample 2 的 rank 是 3
    #
    # 所以：
    #   proj_rank[:, m] = [2, 3, 1]
    proj_rank = zeros(Int, size(X, 2), length(projections))

    # TODO: For weighted samples maintain cumsum of weights instead of sample rank
    #
    # 当前代码假设所有 Dirac samples 权重相同，都是 1/N
    #
    # 所以 empirical CDF 可以直接用 rank:
    #
    # F_D(x_i) ≈ (rank_i - 0.5) / N
    #
    # 如果以后支持 weighted samples，则不能只用 rank，
    # 而应该按照排序后的权重做 cumulative sum。

    # 初始化所有方向上的投影值、排序 permutation 和 rank
    for (i, dir) in enumerate(directions)
        # 计算所有 samples 在第 i 个方向 dir 上的投影
        #
        # X 的大小是 dim × N
        # dir 的大小是 dim
        #
        # 数学上：
        #
        # proj_X[:, i] = X' * dir
        #
        # 也就是：
        #
        # proj_X[j, i] = dir' * X[:, j]
        #
        # 第 j 个 sample 在方向 dir 上的投影值
        #
        # 这里用 mul! 做原地矩阵乘法，避免临时数组
        #
        # @view(proj_X[:, i:i])' 的形状可以理解为 1 × N
        # dir' 的形状是 1 × dim
        # X 的形状是 dim × N
        #
        # 所以：
        #
        # dir' * X 是 1 × N
        #
        # 再写入 proj_X[:, i] 对应的位置
        mul!(@view(proj_X[:, i:i])', dir', X)

        # 对第 i 个方向上的投影值排序
        #
        # sortperm!(dest, values)
        #
        # 会把排序后的索引写入 dest
        #
        # @view(proj_sp[:, i]) 是目标存储位置
        # @view(proj_X[:, i]) 是要排序的投影值
        #
        # 结果：
        #   proj_sp[:, i] 保存按 proj_X[:, i] 从小到大排序后的 sample index
        sortperm!(@view(proj_sp[:, i]), @view(proj_X[:, i]))

        # 计算 inverse permutation
        #
        # proj_sp[:, i]:
        #   rank -> sample index
        #
        # invperm(proj_sp[:, i]):
        #   sample index -> rank
        #
        # 然后保存到 proj_rank[:, i]
        proj_rank[:, i] .= invperm(@view(proj_sp[:, i]))
    end

    # 记录迭代次数
    iters = 0

    # 主迭代循环
    #
    # stop_condition(delta_x) 返回 true 时停止
    #
    # 注意：
    #   delta_x 初始是全 1，所以第一次通常不会停止
    #
    # 每轮迭代大致做：
    #
    # 1. 更新所有 sample 在所有方向上的投影值
    # 2. 更新每个方向上的排序和 rank
    # 3. 根据 use_local 选择更新方法
    # 4. 更新 sample 位置 X
    # 5. iters += 1
    while !stop_condition(delta_x)

        # 并行遍历每个投影方向
        #
        # eachindex(directions) 理论上表示所有方向的索引
        #
        # 每个任务负责一个方向：
        #   - 重新计算 proj_X[:, i]
        #   - 根据新的投影值更新排序 permutation proj_sp[:, i]
        #   - 同步更新 proj_rank[:, i]
        @tasks for i in eachindex(directions)
            @set ntasks=nthreads           

            # 重新计算所有 samples 在第 i 个方向上的投影
            #
            # proj_X[:, i] = directions[i]' * X
            #
            # @view(proj_X[:, i])' 是 1 × N
            # directions[i]' 是 1 × dim
            # X 是 dim × N
            mul!(@view(proj_X[:, i])', directions[i]', X)

            # 取出第 i 个方向对应的排序 permutation
            sp = @view(proj_sp[:, i])

            # 根据新的 proj_X，对排序 permutation 做局部修正
            #
            # 这里没有重新完整 sortperm!
            # 而是基于上一轮的排序结果，做类似 insertion-sort 的局部调整
            #
            # 这样做的原因是：
            #   每次迭代 sample 位置通常只移动一点
            #   所以排序不会发生巨大变化
            #   用局部交换比每轮完整排序更快
            for j in eachindex(sp)

                # k 表示当前元素往前移动了多少步
                #
                # 初始 k = 0
                # 后面每交换一次，k -= 1
                #
                # 所以 j+k 表示当前正在检查的位置
                k = 0

                # 只要当前排序中相邻两个元素顺序不对，就交换它们
                #
                # 条件解释：
                #
                # j+k < length(sp)
                #   确保可以访问 sp[j+k+1]
                #
                # j+k > 0
                #   确保当前索引合法
                #
                # proj_X[sp[j+k], i] > proj_X[sp[j+k+1], i]
                #   如果前一个 sample 的投影值大于后一个，
                #   说明排序顺序错了，需要交换
                while j+k < length(sp) && j+k > 0 && proj_X[sp[j+k], i] > proj_X[sp[j+k+1], i]
                    
                    # 交换相邻两个 sample index
                    sp[j+k], sp[j+k+1] = sp[j+k+1], sp[j+k]

                    # 交换之后，更新这两个 sample 的 rank
                    #
                    # 注意这里的 sp[j+k] 和 sp[j+k+1]
                    # 已经是交换之后的 sample index
                    #
                    # sp[j+k] 被交换到了更靠前的位置，所以 rank 减 1
                    proj_rank[sp[j+k], i] -= 1

                    # sp[j+k+1] 被交换到了更靠后的位置，所以 rank 加 1
                    proj_rank[sp[j+k+1], i] += 1

                    # 继续向前检查
                    #
                    # 如果某个 sample 一直需要往前移动，
                    # k 会变成 -1, -2, -3, ...
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

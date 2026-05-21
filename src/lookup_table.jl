"""
定义 LookupTable 结构体

这个结构体用于保存某个一维分布的 pdf 和 cdf 的查找表

它主要用于加速后续反复计算：
1. target PDF
2. target CDF

在 PCD 的 CvM 梯度/Hessian 计算中，经常需要在很多 x 位置查询目标分布的 pdf/cdf
如果每次都调用 Distributions.pdf 和 Distributions.cdf，可能会比较慢
所以这里提前离散化存表
"""

struct LookupTable
    minv::Float64           # lookup table 覆盖区间的左边界
    maxv::Float64           # lookup table 覆盖区间的右边界
    h::Float64              # grid spacing，也就是表格中相邻采样点之间的间距
    vals::Matrix{Float64}   # 保存 pdf 和 cdf 值的矩阵
                                # vals 的大小是 2 × N
                                # vals[1, :] 保存 pdf 值
                                # vals[2, :] 保存 cdf 值
end

"""
对 LookupTable 版本的目标分布计算 CvM 中需要的 c 和 p

lut: LookupTable
x: 当前 Dirac sample 投影到某个方向后的标量位置
i: 当前样本在该方向投影后排序中的序号
N: 样本总数

返回：
c: 当前 Dirac CDF 和目标 CDF 的差值
p: 目标分布在 x 处的 PDF
"""
function cvm_grad_hess(lut::LookupTable, x, i, N)
    # Dirac mixture 的 empirical CDF 在第 i 个排序点处的近似值
    # 这里使用 (i - 0.5) / N，而不是 i / N
    # 是为了取阶梯函数跳跃位置的中点，常见于 CvM 距离的离散近似
    dirac_cdf = (i-0.5)/N

    if x <= lut.minv        # 如果 x 落在 lookup table 左边界之外
        return dirac_cdf, 1e-6  # 在极左侧，目标 CDF 近似为 0
                                # 所以 c = dirac_cdf - 0 = dirac_cdf
                                #
                                # p 返回一个很小的正数 1e-6
                                # 避免后续 Hessian 或除法相关计算中出现 0
    elseif x >= lut.maxv    # 如果 x 落在 lookup table 右边界之外
        return dirac_cdf-1.0, 1e-6      # 在极右侧，目标 CDF 近似为 1
                                        # 所以 c = dirac_cdf - 1
                                        #
                                        # p 同样返回一个很小的正数
    end

    # 计算 x 落在哪个 lookup table 区间
    #
    # lut.minv 是左边界
    # lut.h 是 grid spacing
    #
    # (x - lut.minv) / lut.h 表示 x 距离左边界有多少个步长
    #
    # ceil(Int, ...) 表示向上取整，得到左侧 grid index
    idx = ceil(Int, (x - lut.minv)/lut.h)

    # 根据 idx 计算该区间左端点 x1
    #
    # 如果 idx = 1，则 x1 = minv
    # 如果 idx = 2，则 x1 = minv + h
    # 以此类推
    x1 = (lut.minv+(idx-1)*lut.h)

    # 对 pdf 做线性插值
    #
    # lut.vals[1, idx]   是左端点的 pdf 值
    # lut.vals[1, idx+1] 是右端点的 pdf 值
    #
    # @inbounds 表示跳过数组边界检查，提高性能
    p = @inbounds lin_int(x, x1, lut.h, lut.vals[1, idx], lut.vals[1, idx+1])

    # 对 cdf 做线性插值
    #
    # lut.vals[2, idx]   是左端点的 cdf 值
    # lut.vals[2, idx+1] 是右端点的 cdf 值
    #
    # c 表示 empirical Dirac CDF 和 target CDF 的差：
    #
    # c = F_Dirac(x) - F_target(x)
    c = dirac_cdf - @inbounds lin_int(x, x1, lut.h, lut.vals[2, idx], lut.vals[2, idx+1])

    c, p    # 返回 CDF 差值 c 和 PDF 值 p
end

# 这个函数根据公式做线性插值
function lin_int(x, x1, h, y1, y2)
    y1 + (x-x1)/h * (y2-y1)
end

#TODO: Better way to select bounds for lut?
"""
对于单峰高斯分布，这还可以。

但是对于一些复杂分布，可能不够好：

重尾分布，比如 Student-t
mean ± 3 std 可能仍然漏掉不少尾部质量。
偏斜分布
左右尾部长度不一样，用对称区间可能不合适。
多峰混合分布
如果 component 离得很远，整体 mean 和 std 可能不稳定，或者中间空白区域占据太多 grid。
bounded distribution
比如 Uniform、Beta，有自然边界，直接用 support 可能更合理。

更稳健的方法通常是用 quantile：

minv = quantile(dist, 1e-6)
maxv = quantile(dist, 1 - 1e-6)

但不是所有 distribution 都有稳定/快速的 quantile。
"""
function create_lut(dist::UnivariateDistribution, N)
    # 用 mean - 3 std 作为左边界
    minv = mean(dist) .- 3*std(dist)
    # 用 mean + 3 std 作为右边界
    maxv = mean(dist) .+ 3*std(dist)
    create_lut(dist, N, minv, maxv)
end
function create_lut(dist::UnivariateMixture, N)
    ms = mean.(components(dist))
    stds = std.(components(dist))
    minv = minimum(ms .- 3*stds)
    maxv = maximum(ms .+ 3*stds)
    create_lut(dist, N, minv, maxv)
end

function create_lut(dist::UnivariateDistribution, N, minv, maxv)
    vs = range(minv, maxv, N)
    h = vs[2] - vs[1]

    v_mat = Matrix{Float64}(undef, 2, N)
    v_mat[1, :] .= Distributions.pdf.(Ref(dist), vs)
    v_mat[2, :] .= Distributions.cdf.(Ref(dist), vs)

    LookupTable(minv, maxv, h, v_mat)
end
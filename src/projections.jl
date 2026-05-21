"""
定义一个 Projections 结构体
它用于保存：
1. 每个投影方向上的目标一维分布 projections
2. 所有投影方向 dirs

P 表示每个 projection target 的类型
    例如 Normal、MixtureModel、LookupTable 等

T 表示方向矩阵中元素的数值类型
    通常是 Float64
"""
struct Projections{P, T}
    # 保存每个方向上的目标投影分布
    # projections[k] 对应第 k 个投影方向上的目标一维分布
    projections::Vector{P}

    # 保存所有投影方向
    # dirs 是一个矩阵，每一列是一个方向向量
    #
    # 如果原分布是 dim 维，一共有 n_directions 个投影方向，
    # 那么 dirs 的大小通常是：
    #
    # dim × n_directions
    #
    # 例如二维中有 3 个方向：
    #
    # dirs = [
    #     d1_x  d2_x  d3_x
    #     d1_y  d2_y  d3_y
    # ]
    dirs::Matrix{T}
end

# 取出 Projections 里面保存的目标投影分布
get_targets(projs::Projections) = projs.projections # potential error, might be get_projs

# 取出所有投影方向
# 注意这里返回的不是整个矩阵，而是 eachcol(projs.dirs)
# 也就是一个“按列遍历方向向量”的迭代器
get_dirs(projs::Projections) = eachcol(projs.dirs)

"""
重载 Base.getindex
给 Projections 定义索引访问方式
这样用户可以写：

projs[i]

得到第 i 个目标投影分布和第 i 个方向

返回值是一个二元组：

(projs.projections[i], @view(projs.dirs[:, i]))

第一个元素：第 i 个目标投影分布
第二个元素：第 i 个投影方向

也就是说，你可以写 target, dir = projs[3]
"""
Base.getindex(projs::Projections, i::Int) = (projs.projections[i], @view(projs.dirs[:, i]))

"""
重载 Base.iterate
这段让 Projections 可以直接用于 for 循环。

也就是说，定义了它以后，你可以写：

for (target, dir) in projs
    # target 是某个方向上的目标一维分布
    # dir 是对应的方向向量
end

这比手动写索引更优雅。
"""
function Base.iterate(projs::Projections, state=0)
    state >= length(projs) && return
    (projs.projections[state+1], @view(projs.dirs[:, state+1])), state+1
end

"""
重载 Base.length
这一行让你可以写：
length(projs)
"""
Base.length(projs::Projections) = length(projs.projections)

# 这个函数生成二维空间里的均匀投影方向。
uniform_directions_2d(n_directions) = ([cos(a), sin(a)] for a in range(0, stop = pi, length = n_directions + 1)[1:end-1])

# 这个函数随机生成 n_directions 个 dim 维方向。
# Muller 方法 
random_directions(dim, n_directions) = normalize!.(eachcol(randn(dim, n_directions)))  # (Muller, 1959)

# 这一段处理的是多维混合分布，比如高斯混合模型 GMM。
function project(dist::MultivariateMixture, u)
    MixtureModel(map(c -> project(c, u), components(dist)), probs(dist))
end

# 对多维高斯分布的处理
project(dist::AbstractMvNormal, u) = Normal(u' * mean(dist), sqrt(u' * cov(dist) * u))

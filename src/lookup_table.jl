struct LookupTable
    minv::Float64
    maxv::Float64
    h::Float64
    vals::Matrix{Float64}
end

function cvm_grad_hess(lut::LookupTable, x, i, N)
    dirac_cdf = (i-0.5)/N
    if x <= lut.minv
        return dirac_cdf, 1e-6
    elseif x >= lut.maxv
        return dirac_cdf-1.0, 1e-6
    end
    idx = ceil(Int, (x - lut.minv)/lut.h)
    x1 = (lut.minv+(idx-1)*lut.h)

    p = @inbounds lin_int(x, x1, lut.h, lut.vals[1, idx], lut.vals[1, idx+1])
    c = dirac_cdf - @inbounds lin_int(x, x1, lut.h, lut.vals[2, idx], lut.vals[2, idx+1])
    c, p
end

function lin_int(x, x1, h, y1, y2)
    y1 + (x-x1)/h * (y2-y1)
end

#TODO: Better way to select bounds for lut?
function create_lut(dist::UnivariateDistribution, N)
    minv = mean(dist) .- 3*std(dist)
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
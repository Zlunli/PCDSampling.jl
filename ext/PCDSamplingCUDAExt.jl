module PCDSamplingCUDAExt
    using PCDSampling
    using CUDA
    using Distributions
    using OhMyThreads

function draw_samples_gpu(dist::MultivariateDistribution, N, dirs; use_local=false, N_lut=128, 
                max_iters=100, eps=1e-6, stop_cond=nothing, init_samples=nothing, verbose=false)
    
    if N_lut < 1
        error("The number of grid points for the lookup tables has to be positive. The GPU version does not support exact evaluation without lookup tables.")
    end

    if !isa(dirs, Matrix)
        dirs = reduce(hcat, dirs)
    end

    luts = prepare_lookup_tables(Float64, dist, dirs, N_lut)

    if isnothing(init_samples)
        init_samples = rand(dist, N)
    end
    if !isa(init_samples, CuArray)
        init_samples = CuArray(init_samples)
    end

    if isnothing(stop_cond)
        stop_cond = max_iters_and_small_delta(max_iters, eps)
    end

    pcd_sample_gpu(luts, CuArray(dirs), init_samples, stop_cond; use_local, verbose)
end

function prepare_lookup_tables(T, dist::MultivariateMixture, directions, N_lut)
    K = size(directions, 2)

    lut_xs_cpu = Matrix{T}(undef, N_lut, K)
    lut_pdf_cpu = Matrix{T}(undef, N_lut, K)
    lut_cdf_cpu = Matrix{T}(undef, N_lut, K)

    @tasks for k in axes(directions, 2)
        proj_target = project(dist, @view(directions[:, k]))
        comps    = components(proj_target)

        μs_k = mean.(comps)
        σs_k = std.(comps)

        μmin, μmax = extrema(μs_k)
        σmax = maximum(σs_k)

        a_k = μmin - 4 * σmax
        b_k = μmax + 4 * σmax

        if a_k == b_k
            a_k -= 1
            b_k += 1
        end
        rs   = range(a_k, b_k; length = N_lut)

        for (i, r) in enumerate(rs)
            lut_xs_cpu[i, k] = r
            lut_pdf_cpu[i, k] = pdf(proj_target, r)
            lut_cdf_cpu[i, k] = cdf(proj_target, r)
        end
    end

    return CuArray(lut_xs_cpu), CuArray(lut_pdf_cpu), CuArray(lut_cdf_cpu)
end

function pcd_sample_gpu(luts, cu_directions::CuArray, init_samples::CuArray, stop_condition; use_local=false, verbose=true)
    K  = size(cu_directions, 2)
    T  = eltype(init_samples)
    d, L = size(init_samples)
    IType = Int32
    K32, d32, L32 = Int32(K), Int32(d), Int32(L)

    Tpb32 = Int32(128)

    cu_positions = init_samples
    cu_projections = CuArray{T}(undef, L, K)

    cu_lut_xs, cu_lut_pdf, cu_lut_cdf = luts

    grad_d = CuArray{T}(undef, L, K)
    delta_d = CuArray{T}(undef, d, L)

    if use_local
        hess_d = nothing
        hess_accum_d = nothing
    else
        hess_d = CuArray{T}(undef, L, K)
        hess_accum_d = CuArray{T}(undef, d, d, L)
    end

    cu_inv_sort_idx = CuArray{IType}(undef, L, K)
    cu_row_idx      = CuArray{IType}(undef, L, K)
    cu_sort_idx32   = CuArray{IType}(undef, L, K)

    # General layout
    blockSize  = (cld(L32, Tpb32), K32)
    threadSize = (Tpb32,)
    
    # Sorting layout
    threads_sort = (1024,)
    blocks_sort  = (K32,)

    # Reduction layout
    warps_per_block = 16
    t_dims_red = (warps_per_block*32,)
    b_dims_grad_red = (ceil(Int, d*L / warps_per_block),)
    nvals_tri = d * (d+1) ÷ 2 * L
    b_dims_hess_red = (ceil(Int, nvals_tri / warps_per_block),)

    # Initialize sorting
    @cuda threads=threadSize blocks=blockSize kernel_compute_radon_projection!(
            L32, d32, K32, cu_positions, cu_directions, cu_projections)
    CUDA.sortperm!(cu_sort_idx32, cu_projections; dims=1)

    norm_cpu = Inf # Make sure to enter loop
    iters = 0
    while !stop_condition(norm_cpu)
        delta_d .= 0

        if iters > 0 # For first iiteration samples are projected and fully sorted outside of the loop
            @cuda threads=threadSize blocks=blockSize kernel_compute_radon_projection!(
                L32, d32, K32, cu_positions, cu_directions, cu_projections)

            @cuda threads=threads_sort blocks=blocks_sort kernel_update_sortperm!(
                cu_projections, cu_sort_idx32, L, K)
            # CUDA.sortperm!(cu_sort_idx32, cu_projections; dims=1)
        end

        @. cu_row_idx = (cu_sort_idx32 - IType(1)) % L32 + IType(1)
        
        @cuda threads=threadSize blocks=blockSize kernel_build_inv_sort_idx!(
            cu_inv_sort_idx, cu_row_idx, L32, K32)

        @cuda threads=threadSize blocks=blockSize compute_grad_hess_lut!(
                L32, K32,
                grad_d, hess_d,
                cu_projections, cu_inv_sort_idx,
                cu_lut_xs, cu_lut_cdf, cu_lut_pdf)

        @cuda threads=t_dims_red blocks=b_dims_grad_red reduce_kernel_grad!(delta_d, grad_d, cu_directions)
    
        if !use_local
            @cuda threads=t_dims_red blocks=b_dims_hess_red reduce_kernel_hess!(hess_accum_d, hess_d, cu_directions)
            
            d_pivot, info, d_LU = CUDA.CUBLAS.getrf_strided_batched!(hess_accum_d, true)
            info2, delta_d = CUDA.CUBLAS.getrs_strided_batched!('N', d_LU, reshape(delta_d, (d, 1, L)), d_pivot)
            delta_d = reshape(delta_d, d, L)
        end

        cu_positions .+= delta_d
        norm_cpu = only(CUDA.maximum(abs.(delta_d)))

        iters += 1
    end

    if verbose
        println("final iteration: $iters")
        println("final delta norm: $(norm_cpu)")
    end

    return Array(cu_positions), iters
end

function kernel_update_sortperm!(proj, sortp, L, K)
    k = blockIdx().x
    if k > K
        return
    end

    nswaps = CuStaticSharedArray(Int32, (1,))

    l = 1000

    offset = blockDim().x
    L_half = div(L, 2)

    @inbounds for _ in 1:l
        swapped = 0
        for tid in threadIdx().x:offset:L_half #even phase
            r = 2*tid
            if r < L  
                if proj[sortp[r, k]] > proj[sortp[r+1, k]]
                    sortp[r, k], sortp[r+1, k] = sortp[r+1, k], sortp[r, k]
                    swapped = 1
                end
            end
        end
        CUDA.sync_threads()

        for tid in threadIdx().x:offset:L_half #odd phase
            r = 2*(tid-1)+1
            if r < L  
                if proj[sortp[r, k]] > proj[sortp[r+1, k]]
                    sortp[r, k], sortp[r+1, k] = sortp[r+1, k], sortp[r, k]
                    swapped = 1
                end
            end
        end
        
        CUDA.@atomic nswaps[1] += swapped
        CUDA.sync_threads()
        
        if nswaps[1] == 0
            return
        end
        CUDA.sync_threads()

        if threadIdx().x == 1
            nswaps[1] = 0
        end
        CUDA.sync_threads()
    end
end

function lower_index(linear_idx::Int, n)
    # Calculate the row using the quadratic formula
    row = floor(Int, (-1 + sqrt(1 + 8 * linear_idx)) / 2)
    # Calculate the column
    col = linear_idx - (row * (row + 1)) ÷ 2
    return (row+1, col+1)
end

function kernel_build_inv_sort_idx!(
    inv_sort_idx::CuDeviceMatrix{I},
    sort_idx    ::CuDeviceMatrix{I},
    L::Int32, K::Int32
) where {I<:Integer}
    r   = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    dir =  blockIdx().y  
    if r > L || dir > K
        return
    end

    @inbounds pt = sort_idx[r, dir] 
    @inbounds inv_sort_idx[pt, dir] = r  
    return
end

function compute_grad_hess_lut!(
    L::Int32, K::Int32,
    grad::CUDA.CuDeviceMatrix{T}, 
    hess::Union{Nothing, CUDA.CuDeviceMatrix{T}},
    projections::CUDA.CuDeviceMatrix{T}, 
    inv_sort_idx::CUDA.CuDeviceMatrix{I}, 
    lut_r::CUDA.CuDeviceMatrix{T},
    lut_cdf::CUDA.CuDeviceMatrix{T},
    lut_pdf::CUDA.CuDeviceMatrix{T}
) where {T<:AbstractFloat, I<:Integer}

    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    k = blockIdx().y

    if i > L || k > K
        return
    end

    N_grid = size(lut_r, 1)
    begin
        r = projections[i, k]

        emp_cdf = 1/L * (inv_sort_idx[i, k] - T(0.5))

        r_min = lut_r[1, k]
        r_max = lut_r[end, k]

        model_cdf = zero(T)
        pdf_val   = zero(T)

        if r <= r_min
            model_cdf = lut_cdf[1, k]
            pdf_val   = lut_pdf[1, k]
        elseif r >= r_max
            model_cdf = lut_cdf[end, k]
            pdf_val   = lut_pdf[end, k]
        else
            inv_dx = (N_grid - 1) / (r_max - r_min)
            t  = (r - r_min) * inv_dx
            j  = floor(Int32, t) + Int32(1)
            if j >= N_grid
                j = N_grid - Int32(1)
            end

            x0 = lut_r[j, k]
            x1 = lut_r[j + 1, k]
            dx = x1 - x0

            if abs(dx) < eps(T)
                model_cdf = lut_cdf[j, k]
                pdf_val   = lut_pdf[j, k]
            else
                α = (r - x0) / dx

                c0 = lut_cdf[j, k]
                c1 = lut_cdf[j + 1, k]
                model_cdf = c0 + α * (c1 - c0)

                p0 = lut_pdf[j, k]
                p1 = lut_pdf[j + 1, k]
                pdf_val = p0 + α * (p1 - p0)
            end
        end

        if isnothing(hess) # local update
            grad[i, k] = (emp_cdf - model_cdf) / (abs(pdf_val) < eps(T) ? eps(T) : pdf_val)
        else # global update
            grad[i, k] = emp_cdf - model_cdf
            hess[i, k] = abs(pdf_val) < eps(T) ? eps(T) : pdf_val
        end
    end
    return
end

function reduce_kernel_hess!(hess, vpdf, proj)
    CUDA.assume(warpsize() == 32)
    D, K = size(proj)
    tid = threadIdx().x
    wid, lane = fldmod1(tid, warpsize())
    warps_per_block = blockDim().x ÷ warpsize()

    vals_per_sample = D * (D+1) ÷ 2
    
    val_idx = ((blockIdx().x-1) * warps_per_block + wid - 1)
    i = (val_idx ÷ vals_per_sample) + 1
    d_idx = mod(val_idx, vals_per_sample)
    d1, d2 = lower_index(d_idx, D)

    if i > size(vpdf, 1)
        return
    end

    val = 0.0
    for k in 1:32:K
        if k+lane-1 <= K
            val += vpdf[i, k+lane-1] * proj[d1, k+lane-1] * proj[d2, k+lane-1]
        else
            val += 0.0
        end
    end
    val = CUDA.reduce_warp(+, val)

    if lane == 1
        hess[d1, d2, i] = val / K
        hess[d2, d1, i] = val / K

    end
    return
end

function reduce_kernel_grad!(grad, vproj_grad, proj)
    CUDA.assume(warpsize() == 32)
    D, K = size(proj)
    tid = threadIdx().x
    wid, lane = fldmod1(tid, warpsize())
    warps_per_block = blockDim().x ÷ warpsize()

    vals_per_sample = D
    
    val_idx = ((blockIdx().x-1) * warps_per_block + wid - 1)
    i = (val_idx ÷ vals_per_sample) + 1
    d_idx = mod(val_idx, vals_per_sample) + 1

    if i > size(vproj_grad, 1)
        return
    end

    val = 0.0
    for k in 1:32:K
        if k+lane-1 <= K
            val += vproj_grad[i, k+lane-1] * proj[d_idx, k+lane-1]
        end
    end
    val = CUDA.reduce_warp(+, val)

    if lane == 1
        grad[d_idx, i] = val / K
    end
    return
end

function kernel_compute_radon_projection!(
    L::Int32, d::Int32, K::Int32,
    positions::CuDeviceMatrix{T},
    directions::CuDeviceMatrix{T},
    projections::CuDeviceMatrix{T},
) where {T<:AbstractFloat}

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = blockIdx().y
    if i > L || k > K
        return
    end

    r = zero(T)
    @inbounds for j in 1:d
        r = muladd(directions[j, k], positions[j, i], r)
    end

    @inbounds projections[i, k] = r
    return
end

end
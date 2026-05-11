using PCDSampling
using CUDA

# using SpecialFunctions
# using IrrationalConstants

function draw_samples_gpu(dist::MultivariateDistribution, N, dirs;
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
    draw_samples_gpu(projections, init_samples; use_local, max_iters, eps, stop_cond, verbose)
end

function draw_samples_gpu(projections::Projections, init_samples; max_iters=100, eps=1e-6, stop_cond=nothing, use_local=false, verbose=false)
    if isnothing(stop_cond)
        stop_cond = max_iters_and_small_delta(max_iters, eps)
    end
    pcd_sample_gpu(projections, init_samples, stop_cond; use_local, verbose)[1]
end

function compute_grad_hess_lut!(
    L::Int32, K::Int32,
    grad::CUDA.CuDeviceMatrix{T}, 
    hess::CUDA.CuDeviceMatrix{T},
    projections::CUDA.CuDeviceMatrix{T}, 
    dm_weights::CUDA.CuDeviceVector{T},
    inv_sort_idx::CUDA.CuDeviceMatrix{I}, 
    lut_r::CUDA.CuDeviceVector{T},
    lut_cdf::CUDA.CuDeviceVector{T},
    lut_pdf::CUDA.CuDeviceVector{T},
    N_grid::Int32,
) where {T<:AbstractFloat, I<:Integer}

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = blockIdx().y  
    if i > L || k > K
        return
    end

    @inbounds begin
        r = projections[i, k]

        emp_cdf = dm_weights[i] * (inv_sort_idx[i, k] - T(0.5))

        base = (k - 1) * N_grid
        r_min = lut_r[base + 1]
        r_max = lut_r[base + N_grid]

        model_cdf = zero(T)
        pdf_val   = zero(T)

        if r <= r_min
            model_cdf = lut_cdf[base + 1]
            pdf_val   = lut_pdf[base + 1]
        elseif r >= r_max
            model_cdf = lut_cdf[base + N_grid]
            pdf_val   = lut_pdf[base + N_grid]
        else
            inv_dx = (T(N_grid - 1)) / (r_max - r_min)
            t  = (r - r_min) * inv_dx
            j  = Int32(floor(t)) + Int32(1)
            if j < 1
                j = 1
            elseif j >= N_grid
                j = N_grid - 1
            end

            x0 = lut_r[base + j]
            x1 = lut_r[base + j + 1]
            dx = x1 - x0

            if abs(dx) < eps(T)
                model_cdf = lut_cdf[base + j]
                pdf_val   = lut_pdf[base + j]
            else
                α = (r - x0) / dx

                c0 = lut_cdf[base + j]
                c1 = lut_cdf[base + j + 1]
                model_cdf = c0 + α * (c1 - c0)

                p0 = lut_pdf[base + j]
                p1 = lut_pdf[base + j + 1]
                pdf_val = p0 + α * (p1 - p0)
            end
        end

        grad[i, k] = emp_cdf - model_cdf
        hess[i, k] = abs(pdf_val) < eps(T) ? eps(T) : pdf_val
    end
    return
end

function compute_grad_hess_lut_local!(
    L::Int32, K::Int32,
    grad_d_hess::CUDA.CuDeviceMatrix{T}, 
    projections::CUDA.CuDeviceMatrix{T}, 
    dm_weights::CUDA.CuDeviceVector{T},
    inv_sort_idx::CUDA.CuDeviceMatrix{I}, 
    lut_r::CUDA.CuDeviceVector{T},
    lut_cdf::CUDA.CuDeviceVector{T},
    lut_pdf::CUDA.CuDeviceVector{T},
    N_grid::Int32,
) where {T<:AbstractFloat, I<:Integer}

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = blockIdx().y  
    if i > L || k > K
        return
    end

    @inbounds begin
        r = projections[i, k]

        emp_cdf = dm_weights[i] * (inv_sort_idx[i, k] - T(0.5))

        base = (k - 1) * N_grid
        r_min = lut_r[base + 1]
        r_max = lut_r[base + N_grid]

        model_cdf = zero(T)
        pdf_val   = zero(T)

        if r <= r_min
            model_cdf = lut_cdf[base + 1]
            pdf_val   = lut_pdf[base + 1]
        elseif r >= r_max
            model_cdf = lut_cdf[base + N_grid]
            pdf_val   = lut_pdf[base + N_grid]
        else
            inv_dx = (T(N_grid - 1)) / (r_max - r_min)
            t  = (r - r_min) * inv_dx
            j  = Int32(floor(t)) + Int32(1)
            if j < 1
                j = 1
            elseif j >= N_grid
                j = N_grid - 1
            end

            x0 = lut_r[base + j]
            x1 = lut_r[base + j + 1]
            dx = x1 - x0

            if abs(dx) < eps(T)
                model_cdf = lut_cdf[base + j]
                pdf_val   = lut_pdf[base + j]
            else
                α = (r - x0) / dx

                c0 = lut_cdf[base + j]
                c1 = lut_cdf[base + j + 1]
                model_cdf = c0 + α * (c1 - c0)

                p0 = lut_pdf[base + j]
                p1 = lut_pdf[base + j + 1]
                pdf_val = p0 + α * (p1 - p0)
            end
        end

        grad_d_hess[i, k] = (emp_cdf - model_cdf) / (abs(pdf_val) < eps(T) ? eps(T) : pdf_val)
    end
    return
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

function prepare_radon_cvm_inputs_lut(
    DM_Approximation, directions_mat, target_dist, L::IType, K::IType, N_grid::IType) where IType
    I = IType
    T = eltype(DM_Approximation.positions)
    

    projections = CuArray{T}(undef, L, K)
    inv_sort_idx = CuArray{I}(undef, L, K)

    positions = CuArray(DM_Approximation.positions)
    
    raw_weights = DM_Approximation.weights
    if isa(raw_weights, AbstractArray)
        dm_weights = CuArray(raw_weights)
    else
        dm_weights = CuArray(fill(T(raw_weights), L))
    end
    
    proj_targets = [radon(target_dist, u) for u in eachcol(directions_mat)]

    directions = CuArray(directions_mat) 

    total_len = K * N_grid
    lut_r_cpu   = Vector{T}(undef, total_len)
    lut_pdf_cpu = Vector{T}(undef, total_len)
    lut_cdf_cpu = Vector{T}(undef, total_len)

    # for (k, proj_target) in enumerate(proj_targets)
    @tasks for k in eachindex(proj_targets)
        proj_target = proj_targets[k]
        comps    = components(proj_target)

        μs_k = mean.(comps)
        σs_k = std.(comps)

        μmin = minimum(μs_k)
        μmax = maximum(μs_k)
        σmax = maximum(σs_k)

        a_k = T(μmin - 4 * σmax)
        b_k = T(μmax + 4 * σmax)

        if a_k == b_k
            a_k -= one(T)
            b_k += one(T)
        end

        rs   = range(a_k, b_k; length = N_grid)
        base = (k - 1) * N_grid
        j    = 1

        @inbounds for r in rs
            idx = base + j
            lut_r_cpu[idx]   = r
            lut_pdf_cpu[idx] = pdf(proj_target, r)
            lut_cdf_cpu[idx] = cdf(proj_target, r)
            j += 1
        end
    end

    lut_r   = CuArray(lut_r_cpu)
    lut_pdf = CuArray(lut_pdf_cpu)
    lut_cdf = CuArray(lut_cdf_cpu)

    return (
        projections, inv_sort_idx,
        positions, dm_weights,
        directions,
        lut_r, lut_pdf, lut_cdf
    )
end

function pcd_global_gpu(target_dist::Distribution,
                                     DM_Approximation,
                                     directions,
                                     stop_condition;
                                     print_final_stop=true)

    directions_mat = reduce(hcat, directions)
    K  = size(directions_mat, 2)
    T  = eltype(DM_Approximation.positions)
    d, L = size(DM_Approximation.positions)
    K32, d32, L32 = Int32(K), Int32(d), Int32(L)

    N_grid32 = Int32(100)
    Tpb32 = Int32(128)

    cu_projections::CuArray{Float64}, cu_inv_sort_idx,
    cu_positions, cu_dm_weights,
    cu_directions,
    cu_lut_r, cu_lut_pdf, cu_lut_cdf = 
    prepare_radon_cvm_inputs_lut(DM_Approximation, directions_mat, target_dist, L32, K32, N_grid32)

    grad_d = CuArray{T}(undef, L, K)
    hess_d = CuArray{T}(undef, L, K)

    grad_accum_d = CuArray{T}(undef, d, L)
    hess_accum_d = CuArray{T}(undef, d, d, L)
    delta_d = similar(grad_accum_d)

    IType = Int32
    cu_row_idx      = CuArray{IType}(undef, L, K)
    cu_sort_idx32   = CuArray{IType}(undef, L, K)

    blocks_proj  = (cld(L32, Tpb32), K32)
    threads_proj = (Tpb32,)

    blocks_inv  = (cld(L32, Tpb32), K32)
    threads_inv = (Tpb32,)
    
    threads_cvm = (Tpb32, Int32(1), Int32(1))
    blocks_cvm  = (cld(L32, Tpb32), K32, Int32(1))

    warps_per_block = 16
    t_dims_red = (warps_per_block*32,)
    b_dims_grad_red = (ceil(Int, d*L / warps_per_block),)
    nvals_tri = d * (d+1) ÷ 2 * L
    b_dims_hess_red = (ceil(Int, nvals_tri / warps_per_block),)


    norm_cpu = Inf # Make sure to enter loop
    iters = 0
    while !stop_condition(norm_cpu)
        delta_d .= 0

        blocks_proj  = (cld(L32, Tpb32), K32)
        threads_proj = (Tpb32,)
        @cuda threads=threads_proj blocks=blocks_proj kernel_compute_radon_projection!(
            L32, d32, K32,
            cu_positions, cu_directions,
            cu_projections,
        )
    
        threads_sort = (1024,)
        blocks_sort  = (K32,)
        @cuda threads=threads_sort blocks=blocks_sort kernel_update_sortperm!(
            cu_projections, cu_sort_idx32, L, K)

        # CUDA.sortperm!(cu_sort_idx32, cu_projections; dims=1)

        @. cu_row_idx = (cu_sort_idx32 - IType(1)) % L32 + IType(1)
        
        threads_inv  = (Tpb32,)
        blocks_inv   = (cld(L32, Tpb32), K32)
        @cuda threads=threads_inv blocks=blocks_inv kernel_build_inv_sort_idx!(
            cu_inv_sort_idx, cu_row_idx, L32, K32)

        threads_cvm = (Tpb32, Int32(1), Int32(1))
        blocks_cvm  = (cld(L32, Tpb32), K32, Int32(1))
        @cuda threads=threads_cvm blocks=blocks_cvm compute_grad_hess_lut!(
            L32, K32,
            grad_d, hess_d,
            cu_projections, cu_dm_weights, cu_inv_sort_idx,
            cu_lut_r, cu_lut_cdf, cu_lut_pdf, N_grid32
        )

        @cuda threads=t_dims_red blocks=b_dims_grad_red reduce_kernel_grad!(grad_accum_d, grad_d, cu_directions)
        @cuda threads=t_dims_red blocks=b_dims_hess_red reduce_kernel_hess!(hess_accum_d, hess_d, cu_directions)
        d_pivot, info, d_LU = CUDA.CUBLAS.getrf_strided_batched!(hess_accum_d, true)
        info2, delta_d = CUDA.CUBLAS.getrs_strided_batched!('N', d_LU, reshape(grad_accum_d, (d, 1, L)), d_pivot)
        delta_d = reshape(delta_d, d, L)

        cu_positions .+= delta_d

        norm_cpu = only(CUDA.maximum(abs.(delta_d)))

        iters += 1
    end

    if print_final_stop
        println("final iteration: $iters")
        println("final delta norm: $(norm_cpu)")
    end

    return Array(cu_positions), iters
end

function pcd_local_gpu(target_dist::Distribution,
                                     DM_Approximation,
                                     directions,
                                     stop_condition;
                                     print_final_stop=true)

    directions_mat = reduce(hcat, directions)
    K  = size(directions_mat, 2)
    T  = eltype(DM_Approximation.positions)
    d, L = size(DM_Approximation.positions)
    K32, d32, L32 = Int32(K), Int32(d), Int32(L)

    N_grid32 = Int32(100)

    Tpb32 = Int32(128)

    cu_projections::CuArray{Float64}, cu_inv_sort_idx,
    cu_positions, cu_dm_weights,
    cu_directions,
    cu_lut_r, cu_lut_pdf, cu_lut_cdf = 
    prepare_radon_cvm_inputs_lut(DM_Approximation, directions_mat, target_dist, L32, K32, N_grid32)

    grad_d_hess = CuArray{T}(undef, L, K)

    delta_d = CuArray{T}(undef, d, L)

    IType = Int32
    cu_row_idx      = CuArray{IType}(undef, L, K)
    cu_sort_idx32   = CuArray{IType}(undef, L, K)

    blocks_proj  = (cld(L32, Tpb32), K32)
    threads_proj = (Tpb32,)
    @cuda threads=threads_proj blocks=blocks_proj kernel_compute_radon_projection!(
                L32, d32, K32, cu_positions, cu_directions, cu_projections)

    CUDA.sortperm!(cu_sort_idx32, cu_projections; dims=1)

    @. cu_row_idx = (IType(cu_sort_idx32) - IType(1)) % L32 + IType(1)

    blocks_inv  = (cld(L32, Tpb32), K32)
    threads_inv = (Tpb32,)
    @cuda threads=threads_inv blocks=blocks_inv kernel_build_inv_sort_idx!(
        cu_inv_sort_idx, cu_row_idx, L32, K32)
    
    threads_cvm = (Tpb32, Int32(1), Int32(1))
    blocks_cvm  = (cld(L32, Tpb32), K32, Int32(1))
    CUDA.@sync @cuda threads=threads_cvm blocks=blocks_cvm compute_grad_hess_lut_local!(
        L32, K32,
        grad_d_hess,
        cu_projections, cu_dm_weights, cu_inv_sort_idx,
        cu_lut_r, cu_lut_cdf, cu_lut_pdf, N_grid32
    )

    warps_per_block = 16
    t_dims_red = (warps_per_block*32,)
    b_dims_grad_red = (ceil(Int, d*L / warps_per_block),)
    @cuda threads=t_dims_red blocks=b_dims_grad_red reduce_kernel_grad!(delta_d, grad_d_hess, cu_directions)

    cu_positions .+= delta_d
    norm_cpu = only(CUDA.maximum(abs.(delta_d)))

    iters = 0

    while !stop_condition(norm_cpu)
        delta_d .= 0

        blocks_proj  = (cld(L32, Tpb32), K32)
        threads_proj = (Tpb32,)
        @cuda threads=threads_proj blocks=blocks_proj kernel_compute_radon_projection!(
            L32, d32, K32,
            cu_positions, cu_directions,
            cu_projections,
        )
    
        threads_sort = (1024,)
        blocks_sort  = (K32,)
        @cuda threads=threads_sort blocks=blocks_sort kernel_update_sortperm!(
            cu_projections, cu_sort_idx32, L, K)

        @. cu_row_idx = (cu_sort_idx32 - IType(1)) % L32 + IType(1)
        
        threads_inv  = (Tpb32,)
        blocks_inv   = (cld(L32, Tpb32), K32)
        @cuda threads=threads_inv blocks=blocks_inv kernel_build_inv_sort_idx!(
            cu_inv_sort_idx, cu_row_idx, L32, K32)

        threads_cvm = (Tpb32, Int32(1), Int32(1))
        blocks_cvm  = (cld(L32, Tpb32), K32, Int32(1))
        @cuda threads=threads_cvm blocks=blocks_cvm compute_grad_hess_lut_local!(
            L32, K32,
            grad_d_hess,
            cu_projections, cu_dm_weights, cu_inv_sort_idx,
            cu_lut_r, cu_lut_cdf, cu_lut_pdf, N_grid32
        )

        @cuda threads=t_dims_red blocks=b_dims_grad_red reduce_kernel_grad!(delta_d, grad_d_hess, cu_directions)

        cu_positions .+= delta_d
        norm_cpu = only(CUDA.maximum(abs.(delta_d)))

        iters += 1
    end

    if print_final_stop
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

# function kernel_compute_radon_projection!(
#     L::Int32,
#     d::Int32,
#     K::Int32,

#     positions::CuDeviceMatrix{T},

#     directions::CuDeviceMatrix{T},
#     projections::CuDeviceMatrix{T},
# ) where {T <: AbstractFloat}
#     dirac_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
#     direction_idx = blockIdx().y

#     if dirac_idx > L || direction_idx > K
#         return
#     end

#     r_proj = zero(T)

#     @inbounds for j in 1:d
#         u_j = directions[j, direction_idx]
#         x_j = positions[j, dirac_idx]
#         r_proj += u_j * x_j
#     end

#     projections[dirac_idx, direction_idx] = r_proj

#     return
# end
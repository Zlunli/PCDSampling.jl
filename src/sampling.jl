function cvm_grad_hess(dist::UnivariateDistribution, x, i, N)
    dirac_cdf = (i-0.5)/N
    dirac_cdf - cdf(dist, x), pdf(dist, x)
end

function newton_step!(X, delta_x, projections, proj_X, proj_rank; nthreads=Threads.nthreads())
    @tasks for i in axes(X, 2)
        @set ntasks=nthreads           
        @local begin
            local_delta = zeros(eltype(X), size(X, 1))
            hess_x = zeros(eltype(X), size(X, 1), size(X, 1))
        end

        local_delta .= 0.0
        hess_x .= 0.0

        @inbounds for (m, (target, dir)) in enumerate(projections)
            step, hess_step = cvm_grad_hess(target, proj_X[i, m], proj_rank[i, m], size(X, 2))
            
            for j in eachindex(local_delta)
                local_delta[j] += dir[j] * step
            end
            
            for j in axes(hess_x, 2)
                for k in j:size(hess_x, 1)
                    hess_x[k, j] += hess_step * dir[j] * dir[k]
                end
            end
        end
        
        hess_x_sym = Symmetric(hess_x, :L)
        fac = cholesky!(hess_x_sym)
        ldiv!(@view(delta_x[:, i]), fac, local_delta)

        @views X[:, i] .+= delta_x[:, i]
    end
end

function local_update!(X, delta_x, projections, proj_X, proj_rank; nthreads=Threads.nthreads())
    @tasks for i in axes(X, 2)
        @set ntasks=nthreads           
        delta_x[:, i] .= 0.0
        @inbounds for (m, (target, dir)) in enumerate(projections)
            step, hess_step = cvm_grad_hess(target, proj_X[i, m], proj_rank[i, m], size(X, 2))
            
            for j in axes(delta_x, 1)
                delta_x[j, i] += dir[j] * (step / max(hess_step, 1e-3))
            end
        end
        delta_x[:, i] ./= length(projections)
        @views X[:, i] .+= delta_x[:, i]
    end
end

function pcd_sample(projections::Projections, init_samples, stop_condition; use_local=false, verbose=true, nthreads=Threads.nthreads())
    X = init_samples
    directions = get_dirs(projections)

    delta_x = ones(eltype(X), size(X))

    proj_X = zeros(eltype(X), size(X, 2), length(projections))
    proj_sp = zeros(Int, size(X, 2), length(projections))
    proj_rank = zeros(Int, size(X, 2), length(projections))

    # TODO: For weighted samples maintain cumsum of weights instead of sample rank
    for (i, dir) in enumerate(directions)
        mul!(@view(proj_X[:, i:i])', dir', X)
        sortperm!(@view(proj_sp[:, i]), @view(proj_X[:, i]))
        proj_rank[:, i] .= invperm(@view(proj_sp[:, i]))
    end

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

        if use_local
            local_update!(X, delta_x, projections, proj_X, proj_rank; nthreads)
        else
            newton_step!(X, delta_x, projections, proj_X, proj_rank; nthreads)
        end
        iters += 1
    end

    if verbose
        println("final iteration: $iters")
        println("final delta norm: $(norm(delta_x))")
    end
    return X, iters
end

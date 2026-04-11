function cvm_grad_hess(dist::UnivariateDistribution, x, i, N)
    dirac_cdf = (i-0.5)/N
    dirac_cdf - cdf(dist, x), pdf(dist, x)
end

function pcd_sample(projections::Projections, init_samples, stop_condition; verbose=true)
    X = init_samples
    directions = directions(projections)
    targets = projections(projections)

    delta_x = ones(eltype(X), size(X)) # make sure we enter loop

    proj_X = zeros(eltype(X), size(X, 2), length(directions))
    proj_sp = zeros(Int, size(X, 2), length(directions))
    proj_rank = zeros(Int, size(X, 2), length(directions))

    for (i, dir) in enumerate(directions)
        mul!(@view(proj_X[:, i:i])', dir', X)
        sortperm!(@view(proj_sp[:, i]), @view(proj_X[:, i]))
        proj_rank[:, i] .= invperm(@view(proj_sp[:, i]))
    end

    iters = 0

    while !stop_condition(delta_x)
        @tasks for i in eachindex(directions)
        # for i in eachindex(directions)
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

        @tasks for i in axes(X, 2)           
            @local begin
                local_delta = zeros(eltype(X), size(X, 1))
                hess_x = zeros(eltype(X), size(X, 1), size(X, 1))
            end

            local_delta .= 0.0
            hess_x .= 0.0

            @inbounds for (m, dir) in enumerate(directions)
                step, hess_step = cvm_grad_hess(targets[m], proj_X[i, m], proj_rank[i, m], size(X, 2))
                
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

        iters += 1
    end

    if verbose
        println("final iteration: $iters")
        println("final delta norm: $(norm(delta_x))")
    end
    return X
end

function pcd_sample_lut(targets, init_samples, directions::AbstractVector, stop_condition; verbose=true, N_lut=100)
    X = init_samples
    
    delta_x = ones(eltype(X), size(X)) # make sure we enter loop

    proj_X = zeros(eltype(X), size(X, 2), length(directions))
    proj_sp = zeros(Int, size(X, 2), length(directions))
    proj_rank = zeros(Int, size(X, 2), length(directions))

    luts = Vector{LookupTable}(undef, length(directions))

    # TODO: For weighted samples maintain cumsum of weights instead of sample rank
    @tasks for i in eachindex(directions)
    # for i in eachindex(directions)
        mul!(@view(proj_X[:, i:i])', directions[i]', X)
        sortperm!(@view(proj_sp[:, i]), @view(proj_X[:, i]))
        proj_rank[:, i] .= invperm(@view(proj_sp[:, i]))
        luts[i] = create_luts(targets[i], N_lut)
    end

    iters = 0

    while !stop_condition(delta_x)
        @tasks for i in eachindex(directions)
        # for i in eachindex(directions)
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

        @tasks for i in axes(X, 2)
        # for i in axes(X, 2)
            @local begin
                local_delta = zeros(eltype(X), size(X, 1))
                hess_x = zeros(eltype(X), size(X, 1), size(X, 1))
            end

            local_delta .= 0.0
            hess_x .= 0.0

            @inbounds for (m, dir) in enumerate(directions)
                step, hess_step = cvm_grad_hess(luts[m], proj_X[i, m], proj_rank[i, m], size(X, 2))
                
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

        iters += 1
    end
    if verbose
        println("final iteration: $iters")
        println("final delta norm: $(norm(delta_x))")
    end
    return X, iters
end


function pcd_sample_local(targets, init_samples, directions::AbstractVector, stop_condition; verbose=true, N_lut=100)
    X = init_samples
    
    delta_x = ones(eltype(X), size(X)) # make sure we enter loop

    proj_X = zeros(eltype(X), size(X, 2), length(directions))
    proj_sp = zeros(Int, size(X, 2), length(directions))
    proj_rank = zeros(Int, size(X, 2), length(directions))

    luts = Vector{LookupTable}(undef, length(directions))

    @tasks for i in eachindex(directions)
        mul!(@view(proj_X[:, i:i])', directions[i]', X)
        sortperm!(@view(proj_sp[:, i]), @view(proj_X[:, i]))
        proj_rank[:, i] .= invperm(@view(proj_sp[:, i]))
        luts[i] = create_luts(targets[i], N_lut)
    end

    iters = 0

    while !stop_condition(delta_x)
        @tasks for i in eachindex(directions)
        # for i in eachindex(directions)
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

        @tasks for i in axes(X, 2)
        # for i in axes(X, 2)
            @local begin
                local_delta = zeros(eltype(X), size(X, 1))
            end
            local_delta .= 0.0
            @inbounds for (m, dir) in enumerate(directions)
                step, hess_step = cvm_grad_hess(luts[m], proj_X[i, m], proj_rank[i, m], size(X, 2))
                
                for j in eachindex(local_delta)
                    local_delta[j] += dir[j] * (step / max(hess_step, 1e-3))
                end
            end
            local_delta ./= length(directions)
            X[:, i] .+= local_delta
            delta_x[:, i] .= local_delta
        end

        iters += 1
    end

    if verbose
        println("final iteration: $iters")
        println("final delta norm: $(norm(delta_x))")
    end
    return X, iters
end

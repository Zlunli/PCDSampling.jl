struct Projections{P, T}
    projections::Vector{P}
    dirs::Matrix{T}
end

projections(projs::Projections) = projs.projections
directions(projs::Projections) = eachcol(projs.dirs)
Base.getindex(projs::Projections, i::Int) = (projs.projections[i], @view(projs.dirs[:, i]))

function Base.iterate(projs::Projections, state=0)
    state >= length(projs) && return
    (projs.projections[state+1], @view(projs.dirs[:, state+1])), state+1
end
Base.length(projs::Projections) = length(projs.projections)

uniform_directions_2d(n_directions) = ([cos(a), sin(a)] for a in range(0, stop = pi, length = n_directions + 1)[1:end-1])
random_directions(dim, n_directions) = normalize!.(eachcol(randn(dim, n_directions)))  # (Muller, 1959)

function project(dist::MultivariateMixture, u)
    MixtureModel(map(c -> project(c, u), components(dist)), probs(dist))
end

project(dist::AbstractMvNormal, u) = Normal(u' * mean(dist), sqrt(u' * cov(dist) * u))

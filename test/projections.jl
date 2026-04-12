@testset "create directions" begin
    dirs = uniform_directions_2d(10)
    @test length(dirs) == 10
    @test length(first(dirs)) == 2

    dirs = random_directions(2, 100)
    @test length(dirs) == 100
    @test length(first(dirs)) == 2

    dirs = random_directions(6, 200)
    @test length(dirs) == 200
    @test length(first(dirs)) == 6
end

@testset "project multivariate" begin
    dist = MvNormal([-1, 2.0], Diagonal([1.0, 2.0]))

    @test project(dist, [1.0, 0.0]) == Normal(-1.0, 1.0)
    @test project(dist, [0.0, 1.0]) == Normal(2.0, sqrt(2))
    @test project(dist, [1/sqrt(2), 1/sqrt(2)]) == Normal(1/sqrt(2), sqrt(3)/sqrt(2))
end

@testset "project mixture" begin
    dist = MixtureModel([MvNormal([0.0, 0.0], I(2)), MvNormal([1.0, 0.0], I(2))], [0.3, 0.7])
    pdist = project(dist, [1/sqrt(2), 1/sqrt(2)])

    ref = MixtureModel([Normal(0.0, 1.0), Normal(1/sqrt(2), 1.0)], [0.3, 0.7])

    @test all(probs(pdist) .== probs(ref))
    
    for (c, r) in zip(components(pdist), components(ref))
        @test mean(c) ≈ mean(r)
        @test std(c) ≈ std(r)
    end
end

@testset "check Projections" begin
    projs = Projections([Normal(1.0, 1.0), Normal(0.0, 1.0)], [1.0 0.0; 0.0 1.0])
    @test length(projs) == 2
    @test projs[1] == (Normal(1.0, 1.0), [1.0, 0.0])
    @test projs[2] == (Normal(0.0, 1.0), [0.0, 1.0])

    c = 0
    for _ in projs
        c += 1
    end
    @test c == 2
end
@testset "lookup table creation" begin
    N = 100
    dist = Normal(0.0, 1.0)
    mix_dist = MixtureModel([Normal(0.0, 1.0), Normal(-1.0, 1.1)])

    lut = create_lut(dist, N)
    @test size(lut.vals) == (2, N)

    lut = create_lut(mix_dist, N)
    @test size(lut.vals) == (2, N)
end

@testset "lookup table evaluation" begin
    mix_dist = MixtureModel([Normal(0.0, 1.0), Normal(-1.0, 1.1)])
    N = 12
    x = -0.4
    i = 3

    lut = create_lut(mix_dist, 100)
    g, h = cvm_grad_hess(lut, x, i, N)
    @test g ≈ (i-0.5)/N - cdf(mix_dist, x)  atol=1e-3
    @test h ≈ pdf(mix_dist, x) atol=1e-3

    lut = create_lut(mix_dist, 10000)
    g, h = cvm_grad_hess(lut, x, i, N)
    @test g ≈ (i-0.5)/N - cdf(mix_dist, x)  atol=1e-8
    @test h ≈ pdf(mix_dist, x) atol=1e-8
end
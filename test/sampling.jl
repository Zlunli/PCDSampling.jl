@testset "cvm_grad_hess" begin
    mix_dist = MixtureModel([Normal(0.0, 1.0), Normal(-1.0, 1.1)])
    N = 12
    x = -0.4
    i = 3

    g, h = cvm_grad_hess(mix_dist, x, i, N)
    @test g ≈ (i-0.5)/N - cdf(mix_dist, x)
    @test h ≈ pdf(mix_dist, x)
end

@testset "draw samples" begin
    dist = MixtureModel([MvNormal([0.0, 0.0], I(2))])
    X = draw_samples(dist, 10, uniform_directions_2d(20); verbose=false)
    @test size(X) == (2, 10)

    X = draw_samples(dist, 10, uniform_directions_2d(20); verbose=false, N_lut=-1)
    @test size(X) == (2, 10)

    X = draw_samples(dist, 10, uniform_directions_2d(20); verbose=false, use_local=true)
    @test size(X) == (2, 10)
end
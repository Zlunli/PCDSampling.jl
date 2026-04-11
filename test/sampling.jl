@testset "cvm_grad_hess" begin
    mix_dist = MixtureModel([Normal(0.0, 1.0), Normal(-1.0, 1.1)])
    N = 12
    x = -0.4
    i = 3

    g, h = cvm_grad_hess(mix_dist, x, i, N)
    @test g ≈ (i-0.5)/N - cdf(mix_dist, x)
    @test h ≈ pdf(mix_dist, x)
end
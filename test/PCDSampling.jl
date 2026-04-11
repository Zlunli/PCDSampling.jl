@testset "stop conditions" begin
    stop = fixed_iters(10)
    c = 0
    while !stop(nothing)
        c += 1
    end
    @test c == 10

    stop = max_iters_and_small_delta(10, 1e-2)
    c = 0
    while !stop(2.0)
        c += 1
    end
    @test c == 10
    
    stop = max_iters_and_small_delta(10, 1e-2)
    c = 0
    eps = [1.0]
    while !stop(eps)
        eps ./= 11
        c += 1
    end
    @test c == 2
end
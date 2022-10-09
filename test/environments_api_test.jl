@testset "Check is environments follow standard API" begin
    test_environments = Dict("InvertedPendulumEnv" => InvertedPendulumEnv)

    for (env_name, env) in test_environments
        @testset "Check basic API for $env_name" begin
            # Check if it starts
            e = @test_nowarn env()
            @test_nowarn reset!(e, 12785)
            @test_nowarn reset!(e)
            as = @test_nowarn action_space(e)
            @test_nowarn step!(e, as[1])
            st = @test_nowarn state(e)
            @test_nowarn setstate!(e, st)
        end

        @testset "Check if state management works as expected for $env_name" begin
            e = env()
            st0 = reset!(e)
            reset!(e)
            st1 = state(e)
            @test st0 ≉ st1
            setstate!(e, st0)
            st2 = state(e)
            @test st0 ≈ st2
        end
    end
end
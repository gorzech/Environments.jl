@testset "Check is environments follow standard API" begin
    test_environments = Dict(
        "InvertedPendulumEnv" => InvertedPendulumEnv,
        "InvertedDoublePendulumEnv" => InvertedDoublePendulumEnv,
    )

    for (env_name, env) in test_environments
        @testset "Check basic API for $env_name" begin
            # Check if it starts
            e = @test_nowarn env()
            @test_nowarn reset!(e, 12785)
            @test_nowarn reset!(e)
            as = @test_nowarn action_space(e)
            st = @test_nowarn state(e)
            @test_nowarn step!(e, as[1])
            @test_nowarn setstate!(e, st)
            @test_nowarn isdone(e, st)
            @test !isdone(e, st)
        end

        @testset "Check if state management works as expected for $env_name" begin
            e = env()
            reset!(e)
            st0 = state(e)
            reset!(e)
            st1 = state(e)
            @test st0.y ≉ st1.y
            @test st0.steps_beyond_terminated === st1.steps_beyond_terminated
            setstate!(e, st0)
            st2 = state(e)
            @test st0.y ≈ st2.y
        end
    end
end
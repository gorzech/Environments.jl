# using Revise
using Environments
using Test

@testset "Environments.jl" begin
    # Write your tests here.
    include("environments_api_test.jl")
end

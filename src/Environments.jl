module Environments
using Random
using StaticArrays
# Write your package code here.
include("abstract_environment.jl")
export AbstractEnvironment
export reset!, step!, state, setstate!, action_space, isdone
include("pendulums/inverted_pendulum.jl")
export InvertedPendulumEnv

end

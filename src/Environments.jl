module Environments
using Random
# Write your package code here.
include("abstract_environment.jl")
export AbstractEnvironment
export reset!, step!, state, setstate!
include("pendulums/inverted_pendulum.jl")
export InvertedPendulumEnv

end

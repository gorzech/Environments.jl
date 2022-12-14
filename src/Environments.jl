module Environments
using Random
using StaticArrays
# Write your package code here.
include("abstract_environment.jl")
export AbstractEnvironment
export reset!, step!, state, setstate!, action_space, isdone, render!
include("pendulums/abstract_inverted_pendulum.jl")
export PendulumOpts
include("pendulums/inverted_pendulum.jl")
export InvertedPendulumEnv
include("pendulums/inverted_double_sim.jl")
include("pendulums/inverted_double_pendulum.jl")
export InvertedDoublePendulumEnv

end

struct PendulumData
    gravity::Float64
    masscart::Float64
    masspole::Float64
    total_mass::Float64
    length::Float64 # actually half the pole's length
    polemass_length::Float64
    force_mag::Float64
    tau::Float64  # seconds between state updates
    kinematics_integrator::String

    # Angle at which to fail the episode
    theta_threshold_radians::Float64
    x_threshold::Float64
end

mutable struct PendulumState{T}
    y::SVector{T,Float64}
    steps_beyond_terminated::Int
    steps::Int
end

PendulumState{T}(y::AbstractVector) where {T} = PendulumState{T}(y, -1, 0)

import Base: copy
copy(ips::PendulumState) = PendulumState(ips.y, ips.steps_beyond_terminated, ips.steps)


const pendulum_max_steps = 200

function is_state_terminated(state, pendulum_data)
    x = state[1]
    thetas = state[3:2:end]
    (
        x < -pendulum_data.x_threshold ||
        x > pendulum_data.x_threshold ||
        any(thetas .< -pendulum_data.theta_threshold_radians) ||
        any(thetas .> pendulum_data.theta_threshold_radians)
    )
end


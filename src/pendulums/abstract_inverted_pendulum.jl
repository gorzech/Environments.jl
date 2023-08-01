struct PendulumOpts
    cart_displacement_penalty::Float64
    # Default angle at which to fail the episode
    theta_threshold_radians::Float64
    x_threshold::Float64
    max_steps::Int
    PendulumOpts(; threshold_factor=1.0, cart_displacement_penalty=0.0, theta_threshold_radians=12 * 2 * pi / 360, x_threshold=2.4, max_steps=200) =
        new(cart_displacement_penalty, threshold_factor * theta_threshold_radians, threshold_factor * x_threshold, max_steps)
end

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
end

mutable struct PendulumState{N}
    y::SVector{N,Float64}
    steps_beyond_terminated::Int
    steps::Int
end

PendulumState{N}(y::AbstractVector) where {N} = PendulumState{N}(y, -1, 0)

import Base: copy
copy(ips::PendulumState) = PendulumState(ips.y, ips.steps_beyond_terminated, ips.steps)

mutable struct PendulumEnv{N} <: AbstractEnvironment
    state::Union{Nothing,PendulumState{N}}

    data::PendulumData
    opts::PendulumOpts
    action_space::SVector{2,Int}
    # render stuff
    screen #::Union{Nothing,Vector{Observable{Vector{Point{2,Float32}}}}}
end

PendulumEnv{N}(data::PendulumData, opts::PendulumOpts=PendulumOpts()) where {N} =
    PendulumEnv{N}(nothing, data, opts, (0, 1), nothing)

pendulum_env_state_size(::PendulumEnv{N}) where {N} = N
pendulum_env_state_size(::PendulumState{N}) where {N} = N

function is_state_terminated(state, pendulum_opts)
    x = state[1]
    thetas = state[3:2:end]
    (
        x < -pendulum_opts.x_threshold ||
        x > pendulum_opts.x_threshold ||
        any(thetas .< -pendulum_opts.theta_threshold_radians) ||
        any(thetas .> pendulum_opts.theta_threshold_radians)
    )
end

mean(y) = sum(y) / length(y)

function reward(env::PendulumEnv) 
    1.0 - env.opts.cart_displacement_penalty * abs(env.state.y[1]) / env.opts.x_threshold
end

function step!(env::PendulumEnv, action::Int)
    @assert !isnothing(env.state) "Call reset before using step function."
    env.state.y = step(env.state.y, action, env.data)
    env.state.steps += 1

    terminated =
        is_state_terminated(env.state.y, env.opts) || env.state.steps >= env.opts.max_steps

    _reward = 0.0
    if !terminated
        _reward = reward(env)
    elseif env.state.steps_beyond_terminated < 0
        # Pole just fell!
        env.state.steps_beyond_terminated = 0
        _reward = reward(env)
    else
        if env.state.steps_beyond_terminated == 0
            @warn (
                "You are calling 'step()' even though this " *
                "environment has already returned terminated = True. You " *
                "should always call 'reset()' once you receive 'terminated = " *
                "True' -- any further steps are undefined behavior."
            )
        end
        env.state.steps_beyond_terminated += 1
        _reward = 0.0
    end

    return (env.state.y, _reward, terminated, nothing)
end

function reset!(env::PendulumEnv, seed::Union{Nothing,Int}=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    low, high = -0.05, 0.05
    N = pendulum_env_state_size(env)
    env.state = PendulumState{N}(rand(Float64, (N,)) .* (high - low) .+ low)
    return env.state.y
end

function isdone(env::PendulumEnv, state::PendulumState)
    return state.steps_beyond_terminated >= 0 || state.steps >= env.opts.max_steps
end

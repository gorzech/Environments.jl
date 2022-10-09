# Code is more or less after the cartpole:
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
struct InvertedPendulumData
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

    InvertedPendulumData() =
        new(9.81, 1.0, 0.1, 1.1, 0.5, 0.05, 10.0, 0.02, "euler", 12 * 2 * pi / 360, 2.4)
end

function step(state, action, p::InvertedPendulumData)
    x, x_dot, theta, theta_dot = state
    force = if action == 1
        p.force_mag
    else
        -p.force_mag
    end
    sintheta, costheta = sincos(theta)

    # For the interested reader:
    # https://coneural.org/florian/papers/05_cart_pole.pdf
    temp = (force + p.polemass_length * theta_dot^2 * sintheta) / p.total_mass
    thetaacc =
        (p.gravity * sintheta - costheta * temp) /
        (p.length * (4.0 / 3.0 - p.masspole * costheta^2 / p.total_mass))
    xacc = temp - p.polemass_length * thetaacc * costheta / p.total_mass

    if p.kinematics_integrator == "euler"
        x = x + p.tau * x_dot
        x_dot = x_dot + p.tau * xacc
        theta = theta + p.tau * theta_dot
        theta_dot = theta_dot + p.tau * thetaacc
    else  # semi-implicit euler
        x_dot = x_dot + p.tau * xacc
        x = x + p.tau * x_dot
        theta_dot = theta_dot + p.tau * thetaacc
        theta = theta + p.tau * theta_dot
    end

    SA[x, x_dot, theta, theta_dot]
end

function is_state_terminated(state, p::InvertedPendulumData)
    x = state[1]
    theta = state[3]
    (
        x < -p.x_threshold ||
        x > p.x_threshold ||
        theta < -p.theta_threshold_radians ||
        theta > p.theta_threshold_radians
    )
end

"""
### Description
This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
    in the left and right direction on the cart.
### Action Space
The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
    of the fixed force the cart is pushed with.
| Num | Action                 |
|-----|------------------------|
| 0   | Push cart to the left  |
| 1   | Push cart to the right |
**Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
    the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it
### Observation Space
The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:
| Num | Observation           | Min                 | Max               |
|-----|-----------------------|---------------------|-------------------|
| 0   | Cart Position         | -4.8                | 4.8               |
| 1   | Cart Velocity         | -Inf                | Inf               |
| 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
| 3   | Pole Angular Velocity | -Inf                | Inf               |
**Note:** While the ranges above denote the possible values for observation space of each element,
    it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
-  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
    if the cart leaves the `(-2.4, 2.4)` range.
-  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
    if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)
### Rewards
Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
including the termination step, is allotted. The threshold for rewards is 475 for v1.
### Starting State
All observations are assigned a uniformly random value in `(-0.05, 0.05)`
### Episode End
The episode ends if any one of the following occurs:
1. Termination: Pole Angle is greater than ±12°
2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
3. Truncation: Episode length is greater than 500 (200 for v0)
### Arguments
```
gym.make('CartPole-v1')
```
No additional arguments are currently supported.
"""
mutable struct InvertedPendulumEnv <: AbstractEnvironment
    state::Union{Nothing,SVector{4, Float64}}

    # This one may be changing
    steps_beyond_terminated::Union{Nothing,Int}

    data::InvertedPendulumData
    action_space::SVector{2,Int}

    InvertedPendulumEnv() = new(nothing, nothing, InvertedPendulumData(), (0, 1))
end

action_space(env::InvertedPendulumEnv) = env.action_space

function step!(env::InvertedPendulumEnv, action::Int)
    @assert !isnothing(env.state) "Call reset before using step function."
    env.state = step(env.state, action, env.data)

    terminated = is_state_terminated(env.state, env.data)

    reward = 0.0
    if !terminated
        reward = 1.0
    elseif isnothing(env.steps_beyond_terminated)
        # Pole just fell!
        env.steps_beyond_terminated = 0
        reward = 1.0
    else
        if env.steps_beyond_terminated == 0
            @warn (
                "You are calling 'step()' even though this " *
                "environment has already returned terminated = True. You " *
                "should always call 'reset()' once you receive 'terminated = " *
                "True' -- any further steps are undefined behavior."
            )
        end
        env.steps_beyond_terminated += 1
        reward = 0.0
    end

    return (env.state, reward, terminated, nothing)
end

function reset!(env::InvertedPendulumEnv, seed::Union{Nothing,Int} = nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    low, high = -0.05, 0.05
    env.state = rand(Float64, (4,)) .* (high - low) .+ low
    env.steps_beyond_terminated = nothing
    return env.state
end

function state(env::InvertedPendulumEnv)
    return env.state
end

function setstate!(env::InvertedPendulumEnv, state)
    env.state = state
    env.steps_beyond_terminated = nothing
    nothing
end
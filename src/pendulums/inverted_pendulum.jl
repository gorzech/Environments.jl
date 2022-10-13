# Code is more or less after the cartpole:
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
InvertedPendulumState = PendulumState{4}

InvertedPendulumData() = PendulumData(
    9.81,
    1.0,
    0.1,
    1.1,
    0.5,
    0.05,
    10.0,
    0.02,
    "euler",
    12 * 2 * pi / 360,
    2.4,
)

function step(state::SVector{4,Float64}, action, p::PendulumData)
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

# ### Description
# This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
# ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
# A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
# The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
#     in the left and right direction on the cart.
# ### Action Space
# The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
#     of the fixed force the cart is pushed with.
# | Num | Action                 |
# |-----|------------------------|
# | 0   | Push cart to the left  |
# | 1   | Push cart to the right |
# **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
#     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it
# ### Observation Space
# The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:
# | Num | Observation           | Min                 | Max               |
# |-----|-----------------------|---------------------|-------------------|
# | 0   | Cart Position         | -4.8                | 4.8               |
# | 1   | Cart Velocity         | -Inf                | Inf               |
# | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
# | 3   | Pole Angular Velocity | -Inf                | Inf               |
# **Note:** While the ranges above denote the possible values for observation space of each element,
#     it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
# -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
#     if the cart leaves the `(-2.4, 2.4)` range.
# -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
#     if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)
# ### Rewards
# Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
# including the termination step, is allotted. The threshold for rewards is 475 for v1.
# ### Starting State
# All observations are assigned a uniformly random value in `(-0.05, 0.05)`
# ### Episode End
# The episode ends if any one of the following occurs:
# 1. Termination: Pole Angle is greater than ±12°
# 2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
# 3. Truncation: Episode length is greater than 500 (200 for v0)
# ### Arguments
# ```
# gym.make('CartPole-v1')
# ```
# No additional arguments are currently supported.

InvertedPendulumEnv = PendulumEnv{4} # 4 is state size

PendulumEnv{4}() = PendulumEnv{4}(InvertedPendulumData())

function state(env::InvertedPendulumEnv)
    if isnothing(env.state)
        nothing
    else
        copy(env.state)
    end
end

function setstate!(env::InvertedPendulumEnv, state)
    env.state = copy(state)
    nothing
end

function isdone(state::InvertedPendulumState)
    return state.steps_beyond_terminated >= 0 || state.steps >= pendulum_max_steps
end

function xycoords(state::InvertedPendulumState, ipd::PendulumData)
    x = state[1]
    θ = state[3]
    cart = [
        Point2f(x - ipd.length, -0.04),
        Point2f(x - ipd.length, 0.04),
        Point2f(x + ipd.length, 0.04),
        Point2f(x + ipd.length, -0.04),
    ]
    x2, y2 = 2ipd.length * [-1, 1] .* sincos(θ)
    pole = [Point2f(x, 0.0), Point2f(x + x2, y2)]
    return cart, pole
end

function render!(env::InvertedPendulumEnv)
    @assert !isnothing(env.state) "Call reset before using render function."
    if isnothing(env.screen)
        c, p = xycoords(env.state.y, env.data)
        cart = Observable(c)
        pole = Observable(p)

        fig = Figure()
        display(fig)
        ax = Axis(fig[1, 1])
        xlims!(ax, -env.data.x_threshold, env.data.x_threshold)
        ylims!(ax, -0.06, 2.1 * env.data.length)

        poly!(ax, cart, color = :grey, strokecolor = :black, strokewidth = 1)
        lines!(ax, pole; linewidth = 12, color = :red)
        scatter!(
            ax,
            pole;
            marker = :circle,
            strokewidth = 2,
            strokecolor = :red,
            color = :red,
            markersize = 36,
        )
        env.screen = [cart, pole]
    end
    # cart and pole observables
    env.screen[1][], env.screen[2][] = xycoords(env.state.y, env.data)
    sleep(0.01)
    nothing
end
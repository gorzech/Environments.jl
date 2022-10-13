InvertedDoublePendulumData() =
    PendulumData(9.81, 1.0, 0.1, 1.1, 0.5, 0.05, 20.0, 0.02, "rk4", 12 * 2 * pi / 360, 2.4)

InvertedDoublePendulumState = PendulumState{6}

function double_inverted(F, a1, a2, a1_t, a2_t, g, l, m, ml)
    t2 = sin(a1)
    t3 = sin(a2)
    t4 = a1 + a2
    t5 = a1 * 2.0
    t6 = a2 * 2.0
    t7 = a1_t^2
    t8 = a2_t^2
    t14 = -a2
    t16 = 1.0 / l
    t17 = m * 4.6e1
    t18 = ml * 4.1e1
    t9 = cos(t5)
    t10 = cos(t6)
    t11 = sin(t5)
    t12 = sin(t6)
    t13 = sin(t4)
    t15 = -t6
    t19 = a1 + t14
    t24 = t5 + t14
    t20 = a1 + t15
    t21 = ml * t10 * 3.0
    t22 = sin(t19)
    t25 = t5 + t15
    t26 = ml * t9 * 2.7e1
    t28 = sin(t24)
    t23 = sin(t20)
    t27 = cos(t25)
    t29 = sin(t25)
    t30 = -t26
    t31 = m * t27 * 1.8e1
    t32 = ml * t27 * 9.0
    t33 = -t31
    t34 = -t32
    t35 = t17 + t18 + t21 + t30 + t33 + t34
    t36 = 1.0 / t35
    x_tt =
        t36 * (
            F * 4.6e1 - F * t27 * 1.8e1 - g * ml * t11 * 2.7e1 +
            g * ml * t12 * 3.0 +
            l * ml * t2 * t7 * 4.5e1 +
            l * ml * t3 * t8 * 5.0 +
            l * ml * t7 * t23 * 3.0 +
            l * ml * t8 * t28 * 9.0
        )
    a1_tt = (
        -t16 *
        t36 *
        (
            F * cos(a1) * 5.4e1 - F * cos(t20) * 1.8e1 - g * m * t2 * 5.4e1 -
            g * m * t23 * 1.8e1 - g * ml * t2 * 8.1e1 - g * ml * t23 * 9.0 +
            l * m * t8 * t22 * 2.4e1 +
            l * m * t7 * t29 * 1.8e1 +
            l * ml * t7 * t11 * 2.7e1 +
            l * ml * t8 * t13 * 9.0 +
            l * ml * t8 * t22 * 2.1e1 +
            l * ml * t7 * t29 * 9.0
        )
    )
    a2_tt = (
        t16 *
        t36 *
        (
            F * cos(a2) * -4.2e1 + F * cos(t24) * 5.4e1 + g * m * t3 * 4.2e1 -
            g * m * t28 * 5.4e1 + g * ml * t3 * 3.0 - g * ml * t28 * 2.7e1 +
            l * m * t7 * t22 * 9.6e1 +
            l * m * t8 * t29 * 1.8e1 +
            l * ml * t7 * t13 * 9.0 +
            l * ml * t8 * t12 * 3.0 +
            l * ml * t7 * t22 * 3.9e1 +
            l * ml * t8 * t29 * 9.0
        )
    )
    return x_tt, a1_tt, a2_tt
end

function integrate_sys_rk4(g, l, m, ml, tau, dt, state, force)
    x, x_t, a1, a1_t, a2, a2_t = state
    function odefun(y)
        x, x_t, a1, a1_t, a2, a2_t = y
        x_tt, a1_tt, a2_tt = double_inverted(force, a1, a2, a1_t, a2_t, g, l, m, ml)
        return SA[x_t, x_tt, a1_t, a1_tt, a2_t, a2_tt]
    end
    yi = @MVector [x, x_t, a1, a1_t, a2, a2_t]
    N_t = tau ÷ dt
    for _ = 1:N_t
        k1 = dt * odefun(yi)
        k2 = dt * odefun(yi + 0.5 * k1)
        k3 = dt * odefun(yi + 0.5 * k2)
        k4 = dt * odefun(yi + k3)

        # Update next value of y
        yi += (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    end
    return SVector{6}(yi)
end

function step(state::SVector{6, Float64}, action, p::PendulumData)
    force = if action == 1
        p.force_mag
    else
        -p.force_mag
    end
    integrate_sys_rk4(
        p.gravity,
        2p.length,
        p.masscart,
        p.masspole,
        p.tau,
        p.tau,
        state,
        force,
    )
end

mutable struct InvertedDoublePendulumEnv <: AbstractEnvironment
    state::Union{Nothing,InvertedDoublePendulumState}

    data::PendulumData
    action_space::SVector{2,Int}

    # render stuff
    screen::Union{Nothing,Vector{Observable{Vector{Point{2,Float32}}}}}

    InvertedDoublePendulumEnv() =
        new(nothing, InvertedDoublePendulumData(), (0, 1), nothing)
end

action_space(env::InvertedDoublePendulumEnv) = env.action_space

function step!(env::InvertedDoublePendulumEnv, action::Int)
    @assert !isnothing(env.state) "Call reset before using step function."
    env.state.y = step(env.state.y, action, env.data)
    env.state.steps += 1

    terminated =
        is_state_terminated(env.state.y, env.data) || env.state.steps >= pendulum_max_steps

    reward = 0.0
    if !terminated
        reward = 1.0
    elseif env.state.steps_beyond_terminated < 0
        # Pole just fell!
        env.state.steps_beyond_terminated = 0
        reward = 1.0
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
        reward = 0.0
    end

    return (env.state.y, reward, terminated, nothing)
end

function reset!(env::InvertedDoublePendulumEnv, seed::Union{Nothing,Int} = nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    low, high = -0.05, 0.05
    env.state = InvertedDoublePendulumState(rand(Float64, (6,)) .* (high - low) .+ low)
    return env.state.y
end

function state(env::InvertedDoublePendulumEnv)
    if isnothing(env.state)
        nothing
    else
        copy(env.state)
    end
end

function setstate!(env::InvertedDoublePendulumEnv, state)
    env.state = copy(state)
    nothing
end

function isdone(state::InvertedDoublePendulumState)
    return state.steps_beyond_terminated >= 0 || state.steps >= pendulum_max_steps
end

function xycoords(state::InvertedDoublePendulumState, ipd::PendulumData)
    x = state[1]
    θ₁ = state[3]
    θ₂ = state[5]
    cart = [
        Point2f(x - ipd.length, -0.04),
        Point2f(x - ipd.length, 0.04),
        Point2f(x + ipd.length, 0.04),
        Point2f(x + ipd.length, -0.04),
    ]
    x2, y2 = [x, 0] + 2ipd.length * [-1, 1] .* sincos(θ₁)
    x3, y3 = [x2, y2] + 2ipd.length * [-1, 1] .* sincos(θ₂)
    pole = [Point2f(x, 0.0), Point2f(x2, y2), Point2f(x3, y3)]
    return cart, pole
end

function render!(env::InvertedDoublePendulumEnv)
    @assert !isnothing(env.state) "Call reset before using render function."
    if isnothing(env.screen)
        c, p = xycoords(env.state.y, env.data)
        cart = Observable(c)
        pole = Observable(p)

        fig = Figure()
        display(fig)
        ax = Axis(fig[1, 1])
        xlims!(ax, -env.data.x_threshold, env.data.x_threshold)
        ylims!(ax, -0.06, 4.1 * env.data.length)
        ax.aspect = DataAspect()

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
    sleep(0.001)
    nothing
end
InvertedDoublePendulumData() =
    PendulumData(9.81, 1.0, 0.1, 1.1, 0.5, 0.05, 20.0, 0.02, "rk4")

InvertedDoublePendulumState = PendulumState{6}

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

function step(state::SVector{6,Float64}, force, p::PendulumData)
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

InvertedDoublePendulumEnv = PendulumEnv{InvertedDoublePendulumState}

PendulumEnv{InvertedDoublePendulumState}() = InvertedDoublePendulumEnv(InvertedDoublePendulumData())
PendulumEnv{InvertedDoublePendulumState}(opts::PendulumOpts) = InvertedDoublePendulumEnv(InvertedPendulumData(), opts)

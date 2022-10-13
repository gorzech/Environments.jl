abstract type AbstractEnvironment end

# Now I must came up with the common Environment API
function close!(env::AbstractEnvironment)
    nothing
end

action_space(env::AbstractEnvironment) = env.action_space

function state(env::AbstractEnvironment)
    if isnothing(env.state)
        nothing
    else
        copy(env.state)
    end
end

function setstate!(env::AbstractEnvironment, state)
    env.state = copy(state)
    nothing
end
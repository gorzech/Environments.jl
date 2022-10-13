abstract type AbstractEnvironment end

# Now I must came up with the common Environment API
function close!(env::AbstractEnvironment)
    nothing
end

action_space(env::AbstractEnvironment) = env.action_space
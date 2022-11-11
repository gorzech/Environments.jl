# Environments

Collection of the Environments (in a form similar to OpenAI gym) that will be used in tests of the Monte Carlo Tree Search (MCTS) algorithm. This is therefore used in connection with the following project: 

[PureMcts.jl](https://github.com/gorzech/PureMCTS.jl)

There are two environments available currently. One is a standard cart pole (single inverted pendulum on the cart). The second is a double inverted pendulum on the car. Both are used to evaluate the MCTS application to mechanical systems.

### Note about *ReinforcementLearning.jl*

There is a plan to make this package compatible with *ReinforcementLearningEnvironment.jl* as shown here [How to write custom environment?](https://juliareinforcementlearning.org/docs/How_to_write_a_customized_environment/).

### Note about *GLMakie.jl*

For visualization of the environment *GLMakie.jl* is used. There is a plan to remove it from the project. The reason is that direct dependency on GLMakie makes it harder to run on headless computers. 

## Environment classes

Currently, two environments are available:

1. `InvertedPendulumEnv`
2. `InvertedDoublePendulumEnv`

They are based on the `AbstractEnvironment` type. As noted above, this may change in the future in favor of compatibility with *ReinforcementLearning.jl*.

## Interface

Currently, the project uses a custom interface. In total, seven methods are available:

1. `reset!`
2. `step!`
3. `action_space`
4. `render!`
5. `isdone`
6. `state`
7. `setstate!`

Methods 1-3 are quite standard, however, they are incomplete for general use with general reinforcement learning methods. At least the `state_space` is missing. 

Method 4 uses *GLMakie.jl* for animating the pendulum. 

Method 5, `isdone`, is useful in connection with the environmental state, not the environment itself. The reason for this is the support of the state management useful for MCTS application. This support is realized with methods 6 and 7: `state` to get the current environment state, and `setstate!` to restore the saved state.

## Acknowledgments

The inverted pendulum is directly inspired by the [Cart Pole](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) environment from OpenAI Gym. It is rewritten version of it with appropriate adjustments. Therefore, most of the parameters are the same as in the original. 
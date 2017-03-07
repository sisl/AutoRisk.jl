export 
    PredefinedBehaviorGenerator,
    UniformBehaviorGenerator,
    rand

"""
# Description:
    - A BehaviorGenerator that selects from a set of predefined behaviors.
"""
type PredefinedBehaviorGenerator <: BehaviorGenerator
    context::ActionContext
    params::Vector{BehaviorParams}
    weights::WeightVec
    rng::MersenneTwister

    """
    # Args:
        - context: context of the drivers
        - params: vector of BehaviorParams
        - weights: the weights associated with the corresponding behaviors,
            i.e., frequency of their random selection (must sum to 1)
        - rng: random number generator
    """
    function PredefinedBehaviorGenerator(context, params, weights, 
            rng = MersenneTwister(1))
        return new(context, params, weights, rng)
    end
end

Base.rand(gen::PredefinedBehaviorGenerator) = sample(gen.params, gen.weights)

"""
# Description:
    - A BehaviorGenerator that generates behavior params uniformly randomly 
        between a minimum and maximum set of params
"""
type UniformBehaviorGenerator <: BehaviorGenerator
    context::ActionContext
    min_p::BehaviorParams
    max_p::BehaviorParams
    rng::MersenneTwister

    """
    # Args:
        - context: context of drivers
        - min_p: the minimum values of params
        - max_p: the maximum values of params
        - rng: random number generator
    """
    function UniformBehaviorGenerator(context, min_p, max_p, 
            rng = MersenneTwister(1))
        return new(context, min_p, max_p, rng)
    end
end

function Base.rand(gen::UniformBehaviorGenerator)
    return rand(gen.rng, gen.min_p, gen.max_p)
end

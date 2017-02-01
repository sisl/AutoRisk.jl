export 
    PredefinedBehaviorGenerator,
    UniformBehaviorGenerator,
    rand

"""
# Description:
    - A BehaviorGenerator that selects from a set of predefined behaviors.
"""
type PredefinedBehaviorGenerator <: BehaviorGenerator
    context::IntegratedContinuous
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
    context::IntegratedContinuous
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


# """
# # Description:
#     - A BehaviorGenerator that generates behavior params from a multivariate
#         normal distribution. 
# """
# type CorrelatedBehaviorGenerator <: BehaviorGenerator
#     context::IntegratedContinuous
#     mus::
#     variances::
#     dist::MvNormal
#     min_p::BehaviorParams
#     max_p::BehaviorParams
#     rng::MersenneTwister

#     """
#     # Args:
#         - context: context of drivers
#         - mu: vector of mean values for the parameters
#         - cov: covariance matrix for all parameters
#         - min_p: the minimum values of params
#         - max_p: the maximum values of params
#         - rng: random number generator
#     """
#     function CorrelatedBehaviorGenerator(context, mus, variances, covmat, 
#             min_p, max_p, rng = MersenneTwister(1))
#         dist = MvNormal(mu, cov)
#         return new(context, dist, min_p, max_p, rng)
#     end
# end

# function Base.rand(gen::CorrelatedBehaviorGenerator)
#     values = rand(gen.rng, gen.dist)
#     return BehaviorParams(
#         IDMParams(values[1:9]...),
#         MOBILParams(values[10:12]...),
#         LateralParams(values[13:end]),
#     )
# end
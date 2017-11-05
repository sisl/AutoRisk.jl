export 
    PredefinedBehaviorGenerator,
    UniformBehaviorGenerator,
    CorrelatedBehaviorGenerator,
    CorrelatedGaussianBehaviorGenerator,
    rand,
    tuncated_gaussian_sample,
    truncated_gaussian_sample_from_agg

"""
# Description:
    - A BehaviorGenerator that selects from a set of predefined behaviors.
"""
type PredefinedBehaviorGenerator <: BehaviorGenerator
    params::Vector{BehaviorParams}
    weights::StatsBase.Weights
    rng::MersenneTwister

    """
    # Args:
        - params: vector of BehaviorParams
        - weights: the weights associated with the corresponding behaviors,
            i.e., frequency of their random selection (must sum to 1)
        - rng: random number generator
    """
    function PredefinedBehaviorGenerator(params, weights, 
            rng = MersenneTwister(1))
        return new(params, weights, rng)
    end
end

Base.rand(gen::PredefinedBehaviorGenerator) = sample(gen.params, gen.weights)

"""
# Description:
    - A BehaviorGenerator that generates behavior params uniformly randomly 
        between a minimum and maximum set of params
"""
type UniformBehaviorGenerator <: BehaviorGenerator
    min_p::BehaviorParams
    max_p::BehaviorParams
    rng::MersenneTwister

    """
    # Args:
        - min_p: the minimum values of params
        - max_p: the maximum values of params
        - rng: random number generator
    """
    function UniformBehaviorGenerator(min_p, max_p, 
            rng = MersenneTwister(1))
        return new(min_p, max_p, rng)
    end
end

function Base.rand(gen::UniformBehaviorGenerator)
    return rand(gen.rng, gen.min_p, gen.max_p)
end

# """
# # Description:
#     - A BehaviorGenerator that generates behavior params by first sampling 
#         aggressiveness, and then deterministically setting the values of the 
#         other parameters conditioned on the aggressiveness s.t. the 
#         aggressiveness interpolates linearly between the min and max values.
# """
# type CorrelatedBehaviorGenerator <: BehaviorGenerator
#     min_p::BehaviorParams
#     max_p::BehaviorParams
#     rng::MersenneTwister
#     """
#     # Args:
#         - min_p: the minimum values of params
#         - max_p: the maximum values of params
#         - rng: random number generator
#     """
#     function CorrelatedBehaviorGenerator(min_p, max_p, 
#             rng = MersenneTwister(1))
#         return new(min_p, max_p, rng)
#     end
# end

# function Base.rand(gen::CorrelatedBehaviorGenerator, agg::Float64=rand(gen.rng))
#     # unpack
#     min_p, max_p = gen.min_p, gen.max_p 

#     # idm
#     idm = IDMParams(
#         min_p.idm.σ + agg * (max_p.idm.σ - min_p.idm.σ),
#         min_p.idm.k_spd + agg * (max_p.idm.k_spd - min_p.idm.k_spd),
#         min_p.idm.δ + agg * (max_p.idm.δ - min_p.idm.δ),
#         max_p.idm.T - agg * (max_p.idm.T - min_p.idm.T), # lower = more agg
#         min_p.idm.v_des + agg * (max_p.idm.v_des - min_p.idm.v_des),
#         max_p.idm.s_min - agg * (max_p.idm.s_min - min_p.idm.s_min), # lower = more agg
#         min_p.idm.a_max + agg * (max_p.idm.a_max - min_p.idm.a_max),
#         min_p.idm.d_cmf + agg * (max_p.idm.d_cmf - min_p.idm.d_cmf), 
#     )

#     # mobil - for all params, lower values indicate more aggressive drivers
#     mobil = MOBILParams(
#         max_p.mobil.politeness - agg * (
#             max_p.mobil.politeness - min_p.mobil.politeness),
#         max_p.mobil.safe_decel - agg * (
#             max_p.mobil.safe_decel - min_p.mobil.safe_decel),
#         max_p.mobil.advantage_threshold - agg * (
#             max_p.mobil.advantage_threshold - min_p.mobil.advantage_threshold)
#     )

#     # lat - for all params, higher values indicate more aggressive drivers
#     lat = LateralParams(
#         min_p.lat.σ + agg * (max_p.lat.σ - min_p.lat.σ),
#         min_p.lat.kp + agg * (max_p.lat.kp - min_p.lat.kp),
#         min_p.lat.kd + agg * (max_p.lat.kd - min_p.lat.kd),
#     )

#     # response time and attentiveness values are identical for all drivers
#     p = BehaviorParams(
#         idm, mobil, lat, 
#         lon_response_time = min_p.lon_response_time,
#         overall_response_time = min_p.overall_response_time,
#         err_p_a_to_i = min_p.err_p_a_to_i,
#         err_p_i_to_a = min_p.err_p_i_to_a,
#     )

#     return p
# end

"""
# Description:
    - A BehaviorGenerator that generates behavior params by first sampling 
        aggressiveness, and then stochastically sampling other parameters 
        conditionally on the aggressiveness value from truncated gaussians.
"""
type CorrelatedGaussianBehaviorGenerator <: BehaviorGenerator
    min_p::BehaviorParams
    max_p::BehaviorParams
    rng::MersenneTwister
    σ::Float64

    """
    # Args:
        - min_p: the minimum values of params
        - max_p: the maximum values of params
        - rng: random number generator
        - sigma: the std dev of truncated gaussian
    """
    function CorrelatedGaussianBehaviorGenerator(
            min_p, 
            max_p, 
            rng = MersenneTwister(1),
            σ = .1)
        return new(min_p, max_p, rng, σ)
    end
end

function tuncated_gaussian_sample(
        rng::MersenneTwister, 
        mu::Float64, 
        σ::Float64,
        high::Float64,
        low::Float64)

    x = mu + randn(rng) * σ
    while x < low || x > high
        x = mu + randn(rng) * σ
    end
    return x
end

function truncated_gaussian_sample_from_agg(
        rng::MersenneTwister,
        agg::Float64,
        relative_σ::Float64,
        high::Float64,
        low::Float64;
        flip::Bool=false)
    
    σ = relative_σ * abs(high - low)
    if flip
        mu = high - agg * (high - low)
        low, high = high, low
    else
        mu = low + agg * (high - low)
    end

    v = tuncated_gaussian_sample(rng, mu, σ, high, low)
    return v
end

function Base.rand(gen::CorrelatedGaussianBehaviorGenerator, agg::Float64=rand(gen.rng))
    # unpack
    min_p = gen.min_p
    max_p = gen.max_p
    rng = gen.rng
    σ = gen.σ

    # idm
    idm = IDMParams(
        truncated_gaussian_sample_from_agg(rng, agg, σ, max_p.idm.σ, min_p.idm.σ),
        truncated_gaussian_sample_from_agg(rng, agg, σ, max_p.idm.k_spd, min_p.idm.k_spd),
        truncated_gaussian_sample_from_agg(rng, agg, σ, max_p.idm.δ, min_p.idm.δ),
        truncated_gaussian_sample_from_agg(rng, agg, σ, max_p.idm.T, min_p.idm.T, flip=true),
        truncated_gaussian_sample_from_agg(rng, agg, σ, max_p.idm.v_des, min_p.idm.v_des),
        truncated_gaussian_sample_from_agg(rng, agg, σ, max_p.idm.s_min, min_p.idm.s_min, flip=true),
        truncated_gaussian_sample_from_agg(rng, agg, σ, max_p.idm.a_max, min_p.idm.a_max),
        truncated_gaussian_sample_from_agg(rng, agg, σ, max_p.idm.d_cmf, min_p.idm.d_cmf)
    )

    # mobil - for all params, lower values indicate more aggressive drivers
    mobil = MOBILParams(
        truncated_gaussian_sample_from_agg(rng, agg, σ, max_p.mobil.politeness, min_p.mobil.politeness, flip=true),
        truncated_gaussian_sample_from_agg(rng, agg, σ, max_p.mobil.safe_decel, min_p.mobil.safe_decel, flip=true),
        truncated_gaussian_sample_from_agg(rng, agg, σ, max_p.mobil.advantage_threshold, min_p.mobil.advantage_threshold, flip=true)
    )

    # lat - for all params, higher values indicate more aggressive drivers
    lat = LateralParams(
        truncated_gaussian_sample_from_agg(rng, agg, σ, max_p.lat.σ, min_p.lat.σ),
        truncated_gaussian_sample_from_agg(rng, agg, σ, max_p.lat.kp, min_p.lat.kp),
        truncated_gaussian_sample_from_agg(rng, agg, σ, max_p.lat.kd, min_p.lat.kd)
    )

    # response time and attentiveness values are identical for all drivers
    p = BehaviorParams(
        idm, mobil, lat, 
        lon_response_time = min_p.lon_response_time,
        overall_response_time = min_p.overall_response_time,
        err_p_a_to_i = min_p.err_p_a_to_i,
        err_p_i_to_a = min_p.err_p_i_to_a,
    )

    return p
end

CorrelatedBehaviorGenerator = CorrelatedGaussianBehaviorGenerator
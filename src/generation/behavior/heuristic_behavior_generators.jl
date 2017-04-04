export 
    PredefinedBehaviorGenerator,
    UniformBehaviorGenerator,
    CorrelatedBehaviorGenerator,
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

"""
# Description:
    - A BehaviorGenerator that generates behavior params by first sampling 
        aggressiveness, and then deterministically setting the values of the 
        other parameters conditioned on the aggressiveness s.t. the 
        aggressiveness interpolates linearly between the min and max values.
"""
type CorrelatedBehaviorGenerator <: BehaviorGenerator
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
    function CorrelatedBehaviorGenerator(context, min_p, max_p, 
            rng = MersenneTwister(1))
        return new(context, min_p, max_p, rng)
    end
end

function Base.rand(gen::CorrelatedBehaviorGenerator)
    # sample aggressiveness
    agg = rand(gen.rng)

    # unpack
    min_p, max_p = gen.min_p, gen.max_p 

    # idm
    idm = IDMParams(
        min_p.idm.σ + agg * (max_p.idm.σ - min_p.idm.σ),
        min_p.idm.k_spd + agg * (max_p.idm.k_spd - min_p.idm.k_spd),
        min_p.idm.δ + agg * (max_p.idm.δ - min_p.idm.δ),
        max_p.idm.T - agg * (max_p.idm.T - min_p.idm.T), # lower = more agg
        min_p.idm.v_des + agg * (max_p.idm.v_des - min_p.idm.v_des),
        max_p.idm.s_min - agg * (max_p.idm.s_min - min_p.idm.s_min), # lower = more agg
        min_p.idm.a_max + agg * (max_p.idm.a_max - min_p.idm.a_max),
        min_p.idm.d_cmf + agg * (max_p.idm.d_cmf - min_p.idm.d_cmf), 
    )

    # mobil - for all params, lower values indicate more aggressive drivers
    mobil = MOBILParams(
        max_p.mobil.politeness - agg * (
            max_p.mobil.politeness - min_p.mobil.politeness),
        max_p.mobil.safe_decel - agg * (
            max_p.mobil.safe_decel - min_p.mobil.safe_decel),
        max_p.mobil.advantage_threshold - agg * (
            max_p.mobil.advantage_threshold - min_p.mobil.advantage_threshold)
    )

    # lat - for all params, higher values indicate more aggressive drivers
    lat = LateralParams(
        min_p.lat.σ + agg * (max_p.lat.σ - min_p.lat.σ),
        min_p.lat.kp + agg * (max_p.lat.kp - min_p.lat.kp),
        min_p.lat.kd + agg * (max_p.lat.kd - min_p.lat.kd),
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


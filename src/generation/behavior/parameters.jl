import Base: rand, ==

export 
    IDMParams,
    MOBILParams,
    LateralParams,
    BehaviorParams,
    uniform,
    get_aggressive_behavior_params,
    get_passive_behavior_params,
    get_normal_behavior_params,
    infer_correlated_aggressiveness

function uniform(rng::MersenneTwister, low::Float64, high::Float64)
    return low + rand(rng) * (high - low)
end

mutable struct IDMParams
    σ::Float64 # standard deviation of action
    k_spd::Float64 # proportional speed tracking constant
    δ::Float64 # acceleration exponent [-]
    T::Float64 # desired time headway [s]
    v_des::Float64 # desired speed [m/s]
    s_min::Float64 # minimum acceptable gap [m]
    a_max::Float64 # maximum acceleration ability [m/s²]
    d_cmf::Float64 # comfortable deceleration [m/s²] (positive)
end

function Base.rand(rng::MersenneTwister, min::IDMParams, 
    max::IDMParams)
    return IDMParams(uniform(rng, min.σ, max.σ),
        uniform(rng, min.k_spd, max.k_spd), 
        uniform(rng, min.δ, max.δ), 
        uniform(rng, min.T, max.T),
        uniform(rng, min.v_des, max.v_des), 
        uniform(rng, min.s_min, max.s_min),
        uniform(rng, min.a_max, max.a_max), 
        uniform(rng, min.d_cmf, max.d_cmf))
end

function Base.:(==)(p1::IDMParams, p2::IDMParams)
    return (p1.σ == p2.σ
            && p1.k_spd == p2.k_spd
            && p1.δ == p2.δ
            && p1.T == p2.T
            && p1.v_des == p2.v_des
            && p1.s_min == p2.s_min
            && p1.a_max == p2.a_max
            && p1.d_cmf == p2.d_cmf)
end

mutable struct MOBILParams
    politeness::Float64 # politeness factor
    safe_decel::Float64 # safe braking value
    advantage_threshold::Float64 # minimum accel
end

function Base.rand(rng::MersenneTwister, min::MOBILParams, 
    max::MOBILParams)
    return MOBILParams(uniform(rng, min.politeness, max.politeness), 
        uniform(rng, min.safe_decel, max.safe_decel), 
        uniform(rng, min.advantage_threshold, max.advantage_threshold))
end

function Base.:(==)(p1::MOBILParams, p2::MOBILParams)
    return (p1.politeness == p2.politeness 
            && p1.safe_decel == p2.safe_decel
            && p1.advantage_threshold == p2.advantage_threshold)
end

mutable struct LateralParams
    σ::Float64 # standard deviation of action
    kp::Float64 # proportional constant for lane tracking
    kd::Float64 # derivative constant for lane tracking
end

function Base.rand(rng::MersenneTwister, min::LateralParams, 
        max::LateralParams)
    return LateralParams(
        uniform(rng, min.σ, max.σ), 
        uniform(rng, min.kp, max.kp), 
        uniform(rng, min.kd, max.kd))
end

function Base.:(==)(p1::LateralParams, p2::LateralParams)
    return p1.σ == p2.σ && p1.kp == p2.kp && p1.kd == p2.kd
end

mutable struct BehaviorParams
    idm::IDMParams
    mobil::MOBILParams
    lat::LateralParams
    lon_response_time::Float64
    overall_response_time::Float64
    err_p_a_to_i::Float64
    err_p_i_to_a::Float64

    function BehaviorParams(idm::IDMParams, mobil::MOBILParams, 
            lat::LateralParams; 
            lon_response_time::Float64 = 0.0,
            overall_response_time::Float64 = 0.0,
            err_p_a_to_i::Float64 = 0.0,
            err_p_i_to_a::Float64 = 0.0)
        return new(idm, mobil, lat, lon_response_time, overall_response_time, 
            err_p_a_to_i, err_p_i_to_a)
    end
end

function Base.:(==)(p1::BehaviorParams, p2::BehaviorParams)
    return (p1.idm == p2.idm && p1.mobil == p2.mobil && p1.lat == p2.lat
        && p1.lon_response_time == p2.lon_response_time
        && p1.overall_response_time == p2.overall_response_time
        && p1.err_p_a_to_i == p2.err_p_a_to_i
        && p1.err_p_i_to_a == p2.err_p_i_to_a)
end

function Base.rand(rng::MersenneTwister, min_p::BehaviorParams, 
        max_p::BehaviorParams)
    idm = rand(rng, min_p.idm, max_p.idm)
    mobil = rand(rng, min_p.mobil, max_p.mobil)
    lat = rand(rng, min_p.lat, max_p.lat)
    lon_response_time = uniform(
        rng, min_p.lon_response_time, max_p.lon_response_time)
    overall_response_time = uniform(
        rng, min_p.overall_response_time, max_p.overall_response_time)
    err_p_a_to_i = uniform(rng, min_p.err_p_a_to_i, max_p.err_p_a_to_i)
    err_p_i_to_a = uniform(rng, min_p.err_p_i_to_a, max_p.err_p_i_to_a)
    return BehaviorParams(idm, mobil, lat, 
        lon_response_time = lon_response_time,
        overall_response_time = overall_response_time,
        err_p_a_to_i = err_p_a_to_i,
        err_p_i_to_a = err_p_i_to_a)
end

# convert params to a vector
function Base.convert(::Type{Vector{Float64}}, p::BehaviorParams)
    values::Vector{Float64} = []
    for f in fieldnames(p)
        fv = getfield(p, f)
        for v in fieldnames(fv)
            push!(values, getfield(fv, v))
        end
    end
    return values
end

""" 
Standard parameter sets
"""

function get_aggressive_behavior_params(;
        lon_σ = 0.0, 
        lat_σ = 0.0, 
        lon_response_time = 0.0,
        overall_response_time = 0.0,
        err_p_a_to_i = 0.0,
        err_p_i_to_a = 0.0)
    return BehaviorParams(
        # IDMParams(lon_σ, 1.5, 4.0, 0.5, 39., 4.0, 4.0, 2.5),
        # IDMParams(lon_σ, 1.5, 4.0, 0.5, 39., 4.0, 4.0, 2.5),
        # IDMParams(lon_σ, 1.5, 4.0, 0.25, 35., 2.0, 3.0, 2.0),
        IDMParams(lon_σ, 1.0, 4.0, .2, 35., 0.0, 6.0, 5.0),
        MOBILParams(0.1, 2.0, 0.01),
        LateralParams(lat_σ, 3.5, 2.5),
        lon_response_time = lon_response_time,
        overall_response_time = overall_response_time,
        err_p_a_to_i = err_p_a_to_i,
        err_p_i_to_a = err_p_i_to_a)
end

function get_passive_behavior_params(;
        lon_σ = 0.0, 
        lat_σ = 0.0, 
        lon_response_time = 0.0,
        overall_response_time = 0.0,
        err_p_a_to_i = 0.0,
        err_p_i_to_a = 0.0)
    return BehaviorParams(
        # IDMParams(lon_σ, 1.0, 4.0, 1.75, 33., 5.0, 1.0, 1.0),
        # IDMParams(lon_σ, 1.0, 4.0, 1.75, 33., 5.0, 1.0, 1.0),
        # IDMParams(lon_σ, 1.0, 4.0, 1., 30., 4.0, 1.0, 1.0),
        IDMParams(lon_σ, .9, 4.0, 1., 25., 4.0, 2.0, 2.0),
        MOBILParams(0.5, 2.0, 0.7),
        LateralParams(lat_σ, 3.0, 2.0),
        lon_response_time = lon_response_time,
        overall_response_time = overall_response_time,
        err_p_a_to_i = err_p_a_to_i,
        err_p_i_to_a = err_p_i_to_a)
end

function get_normal_behavior_params(;
        lon_σ = 0.0, 
        lat_σ = 0.0, 
        lon_response_time = 0.0,
        overall_response_time = 0.0,
        err_p_a_to_i = 0.0,
        err_p_i_to_a = 0.0)
    return BehaviorParams(
        IDMParams(lon_σ, 1.25, 4.0, 1.25, 30., 4.5, 2.5, 1.75),
        MOBILParams(0.3, 2.0, 0.2),
        LateralParams(lat_σ, 3.25, 2.25),
        lon_response_time = lon_response_time,
        overall_response_time = overall_response_time,
        err_p_a_to_i = err_p_a_to_i,
        err_p_i_to_a = err_p_i_to_a)
end

# infer the aggressiveness assuming it was generated from the correlated
# model from the politeness
function infer_correlated_aggressiveness(politeness::Float64, 
        passive_politeness::Float64=.5, aggressive_politeness::Float64=.1)
    return ((aggressive_politeness - politeness) 
        / (aggressive_politeness - passive_politeness))
end
function infer_correlated_aggressiveness(politeness::Array{Float64}, 
        passive_politeness::Float64=.5, aggressive_politeness::Float64=.1)
    return ((aggressive_politeness .- politeness) 
        ./ (aggressive_politeness - passive_politeness))
end
